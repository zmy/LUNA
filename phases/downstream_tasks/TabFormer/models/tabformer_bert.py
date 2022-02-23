import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers.models.bert.modeling_bert import ACT2FN
from transformers.models.bert.modeling_bert import BertForMaskedLM

from .custom_criterion import CustomAdaptiveLogSoftmax

BertLayerNorm = torch.nn.LayerNorm


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.gelu = nn.GELU()

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = self.gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class TabFormerBertConfig(BertConfig):
    def __init__(
            self,
            flatten=True,
            ncols=12,
            vocab_size=30522,
            field_hidden_size=64,
            hidden_size=768,
            num_attention_heads=12,
            pad_token_id=0,
            use_numtok=False,
            use_replace=False,
            use_reg_loss=False,
            data_type=None,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.flatten = flatten
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.use_numtok = use_numtok
        self.use_replace = use_replace
        self.use_reg_loss = use_reg_loss
        self.data_type = data_type


class TabFormerBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.field_hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TabFormerBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TabFormerBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

        self.reg_head = FFNLayer(config.field_hidden_size, config.hidden_size, 2, 0.1)

    def forward(self, hidden_states):
        reg_pred = self.reg_head(hidden_states)
        hidden_states = self.transform(hidden_states)
        mlm_pred = self.decoder(hidden_states)

        return mlm_pred, reg_pred


class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TabFormerBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, vocab):
        super().__init__(config)
        self.use_reg_loss = config.use_reg_loss
        self.data_type = config.data_type
        self.vocab = vocab
        self.cls = TabFormerBertOnlyMLMHead(config)
        self.reg_func = nn.MSELoss(reduction='none')
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            number_values=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]
        bsz, window_size = sequence_output.size(0), sequence_output.size(1)
        if masked_lm_labels is None:
            # If masked_lm_labels not given, use the model as a feature extractor and return the row embedding.
            return sequence_output

        if not self.config.flatten:
            output_sz = list(sequence_output.size())
            expected_sz = [output_sz[0], output_sz[1] * self.config.ncols, -1]
            sequence_output = sequence_output.view(expected_sz)

        prediction_scores, regression_scores = self.cls(sequence_output)  # [bsz * seqlen * vocab_sz]

        outputs = (prediction_scores,) + outputs[2:]

        # prediction_scores : [bsz x seqlen x vsz]
        # masked_lm_labels  : [bsz x seqlen]

        total_masked_lm_loss = 0
        if self.use_reg_loss:
            if self.data_type == 'card':
                regression_scores = regression_scores.view(
                    bsz, window_size, self.config.ncols, 2)[:, :, [2, 3, 8, 9]]
                number_values = number_values.view(
                    bsz, window_size, 4, 1)
            elif self.data_type == 'prsa':
                regression_scores = regression_scores.view(
                    bsz, window_size, self.config.ncols, 2)[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
                number_values = number_values.view(
                    bsz, window_size, 10, 1)

            log_value = torch.log(1e-7 + torch.abs(number_values))
            sgn_value = torch.sgn(number_values)
            value_target = torch.cat((log_value, sgn_value), axis=-1)
            mlm_reg_loss = self.reg_func(regression_scores, value_target).mean(-1)  # bs,win,4
            if self.data_type == 'card':
                mask_indice = masked_lm_labels[:, :, [2, 3, 8, 9]] != -100  # bs,win,4
            elif self.data_type == 'prsa':
                mask_indice = masked_lm_labels[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] != -100  # bs,win,4
            mlm_reg_loss = (mlm_reg_loss * mask_indice).sum(0).sum(0) / (mask_indice.sum(0).sum(0) + 1e-7)  # 4
            mlm_reg_loss = mlm_reg_loss.sum()
            if math.isnan(mlm_reg_loss.item()):
                mlm_reg_loss = torch.tensor(0.0).to(number_values)
            total_masked_lm_loss = total_masked_lm_loss + mlm_reg_loss

        if not self.config.flatten:
            masked_lm_labels = masked_lm_labels.view(expected_sz[0], -1)

        seq_len = prediction_scores.size(1)
        # TODO : remove_target is True for card
        field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=False)
        for field_idx, field_name in enumerate(field_names):
            col_ids = list(range(field_idx, seq_len, len(field_names)))
            global_ids_field = self.vocab.get_field_ids(field_name)

            prediction_scores_field = prediction_scores[:, col_ids, :][:, :, global_ids_field]  # bsz * 10 * K
            masked_lm_labels_field = masked_lm_labels[:, col_ids]
            masked_lm_labels_field_local = self.vocab.get_from_global_ids(global_ids=masked_lm_labels_field,
                                                                          what_to_get='local_ids')

            nfeas = len(global_ids_field)
            loss_fct = self.get_criterion(field_name, nfeas, prediction_scores.device)

            masked_lm_loss_field = loss_fct(prediction_scores_field.view(-1, len(global_ids_field)),
                                            masked_lm_labels_field_local.view(-1))

            total_masked_lm_loss = total_masked_lm_loss + masked_lm_loss_field

        return (total_masked_lm_loss,) + outputs

    def get_criterion(self, fname, vs, device, cutoffs=False, div_value=4.0):

        if fname in self.vocab.adap_sm_cols:
            if not cutoffs:
                cutoffs = [int(vs / 15), 3 * int(vs / 15), 6 * int(vs / 15)]

            criteria = CustomAdaptiveLogSoftmax(in_features=vs, n_classes=vs, cutoffs=cutoffs, div_value=div_value)

            return criteria.to(device)
        else:
            return CrossEntropyLoss()


class TabFormerBertModel(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        return sequence_output
