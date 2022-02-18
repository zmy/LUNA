'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_max
from .utils import spearmanr

from number_encoder.numbed import NumBed, NumBedConfig
from number_tokenizer.numtok import NumTok
from phases.numerical_field.models.table_bert import TableBertModel


class TabertNum(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.momentum = config['momentum']
        self.mlm_probability = config['mlm_probability']
        self.encoder = TableBertModel.from_pretrained(
            'data/ckpt/tabert_origin/tabert_base_k3/model.bin',
        )
        self.reg_head = nn.Linear(768, 2)
        self.distrib_head = nn.Linear(768, 8)
        self.rank_head = nn.Linear(768, 1)

        # create momentum models
        self.encoder_m = TableBertModel.from_pretrained(
            'data/ckpt/tabert_origin/tabert_base_k3/model.bin',
        )
        self.model_pairs = [[self.encoder, self.encoder_m]]
        self.copy_params()
        number_model_config = NumBedConfig(model_name='CharLSTM_base',checkpoint_path='data/ckpt/numbed-ckpt/CharLSTM_base.pt')
        self.number_model = NumBed(number_model_config)
        self.number_proj = nn.Sequential()
        self.loss_ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(self,
                input_ids: torch.Tensor, segment_ids: torch.Tensor,
                context_token_positions: torch.Tensor, column_token_position_to_column_ids: torch.Tensor,
                sequence_mask: torch.Tensor, context_token_mask: torch.Tensor, table_mask: torch.Tensor,
                number_list, table_distributions
                ):
        # ===prepare input===
        raw_input_ids = input_ids
        raw_segment_ids = segment_ids

        batch_size, max_row_num, sequence_len = input_ids.size()
        input_ids = input_ids.reshape(batch_size * max_row_num, -1)
        segment_ids = segment_ids.reshape(batch_size * max_row_num, -1)
        sequence_mask = sequence_mask.reshape(batch_size * max_row_num, -1)

        tokenized = NumTok.tokenize([(x[0], None, None) for x in number_list], input_ids.device)
        num_embedding = self.number_model(tokenized)
        num_embedding = self.number_proj(num_embedding)
        num_embedding = torch.cat((torch.zeros((1, 768)).to(num_embedding), num_embedding), axis=0)

        word_embedding = self.encoder.bert.embeddings.word_embeddings(input_ids)
        num_id = torch.zeros_like(input_ids)
        num_indice = input_ids == self.encoder.tokenizer.num_token_id
        if len(number_list) != 0:
            num_id[num_indice] = torch.range(1, len(number_list)).to(input_ids)
        num_embedding = F.embedding(num_id, num_embedding)
        input_embedding = word_embedding + num_embedding
        mask_embedding, masked_indice = self.mask(input_ids, input_embedding, raw_segment_ids,
                                                  self.encoder.bert.embeddings.word_embeddings)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            context_encoding_m, _, final_table_encoding_m = self.encoder_m.forward_embeds(
                mask_embedding, segment_ids, sequence_mask, max_row_num, column_token_position_to_column_ids,
                table_mask,
                context_token_positions, context_token_mask)
            context_encoding_logits_m = self.encoder_m._bert_model.cls.predictions(context_encoding_m)
            final_table_encoding_logits_m = self.encoder_m._bert_model.cls.predictions(final_table_encoding_m)

        context_encoding, _, final_table_encoding = self.encoder.forward_embeds(
            mask_embedding, segment_ids, sequence_mask, max_row_num, column_token_position_to_column_ids, table_mask,
            context_token_positions, context_token_mask)
        context_encoding_logits = self.encoder._bert_model.cls.predictions(context_encoding)
        final_table_encoding_logits = self.encoder._bert_model.cls.predictions(final_table_encoding)

        _, schema_encoding, schema_encoding_full = self.encoder.forward_embeds(
            input_embedding, segment_ids, sequence_mask, max_row_num, column_token_position_to_column_ids, table_mask,
            context_token_positions, context_token_mask)
        # ===distillation loss===
        max_column_num = table_mask.size(-1)
        table_indice = scatter_max(
            src=masked_indice.to(torch.int64),
            index=column_token_position_to_column_ids.reshape(batch_size * max_row_num, -1),
            dim=-1,  # over `sequence_len`
            dim_size=max_column_num + 1  # last dimension is the used for collecting unused entries
        )[0].to(torch.bool)
        table_indice = table_indice[:, :-1] & (table_mask.reshape(batch_size * max_row_num, -1) > 0.5)
        table_indice = table_indice.reshape(batch_size, max_row_num, -1)
        raw_masked_indice = masked_indice.reshape(batch_size, max_row_num, sequence_len)
        content_indice = raw_masked_indice[:, 0, :context_encoding_m.size(1)] & (
                    raw_segment_ids[:, 0, :context_encoding_m.size(1)] == 0)

        table_soft_label = F.softmax(final_table_encoding_logits_m[table_indice], dim=-1)  # K,C
        table_logits = final_table_encoding_logits[table_indice]  # K,C
        table_loss_distill = -torch.sum(F.log_softmax(table_logits, dim=-1) * table_soft_label, dim=-1).mean()

        content_soft_label = F.softmax(context_encoding_logits_m[content_indice], dim=-1)  # K,C
        content_logits = context_encoding_logits[content_indice]  # K,C
        content_loss_distill = -torch.sum(F.log_softmax(content_logits, dim=-1) * content_soft_label, dim=-1).mean()

        if math.isnan(table_loss_distill.item()):
            table_loss_distill = torch.tensor(0.0).to(input_embedding)
        if math.isnan(content_loss_distill.item()):
            content_loss_distill = torch.tensor(0.0).to(input_embedding)
        loss_distill = (table_loss_distill + content_loss_distill) / 2.0

        # ===mlm loss===

        value_list = torch.tensor([0] + [x[1] for x in number_list]).to(num_embedding)
        value_target0 = F.embedding(num_id, value_list)

        value_target = scatter_max(
            src=value_target0,
            index=column_token_position_to_column_ids.reshape(batch_size * max_row_num, -1),
            dim=-1,  # over `sequence_len`
            dim_size=max_column_num + 1  # last dimension is the used for collecting unused entries
        )[0]
        value_target_full = value_target[:, :-1] * table_mask.reshape(batch_size * max_row_num, -1)
        value_target = value_target_full.reshape(batch_size, max_row_num, -1)[table_indice]

        log_value = torch.log(1e-7 + torch.abs(value_target))
        sgn_value = torch.sgn(value_target)
        value_target = torch.stack((log_value, sgn_value)).T
        value_pred = self.reg_head(final_table_encoding[table_indice])  # K,2
        table_mlm_reg_loss = ((value_pred - value_target) ** 2).mean()
        if math.isnan(table_mlm_reg_loss.item()):
            table_mlm_reg_loss = torch.tensor(0.0).to(input_embedding)

        value_target = value_target0.reshape(batch_size, max_row_num, -1)[:, 0, :context_encoding_m.size(1)]
        is_number = num_id.reshape(batch_size, max_row_num, -1)[:, 0,
                    :context_encoding_m.size(1)] == self.encoder.tokenizer.num_token_id
        value_target = value_target[content_indice & is_number]
        log_value = torch.log(1e-7 + torch.abs(value_target))
        sgn_value = torch.sgn(value_target)
        value_target = torch.stack((log_value, sgn_value)).T
        value_pred = self.reg_head(context_encoding[content_indice & is_number])  # K,2
        content_mlm_reg_loss = ((value_pred - value_target) ** 2).mean()
        if math.isnan(content_mlm_reg_loss.item()):
            content_mlm_reg_loss = torch.tensor(0.0).to(input_embedding)
        language_target = raw_input_ids[:, 0, :context_encoding_m.size(1)][content_indice & ~is_number]  # K
        language_logits = context_encoding_logits[content_indice & ~is_number]  # K,C
        content_mlm_cla_loss = self.loss_ce(language_logits, language_target)
        if math.isnan(content_mlm_cla_loss.item()):
            content_mlm_cla_loss = torch.tensor(0.0).to(input_embedding)
        content_mlm_loss = (content_mlm_reg_loss + content_mlm_cla_loss) / (
                    value_target.size(0) + language_target.size(0) + 1e-7)
        loss_mlm = (table_mlm_reg_loss + content_mlm_loss) / 2.0

        # ===distribution===
        num_sel_cols = min(max_column_num, 50)
        table_indice = table_mask[:, 0, :num_sel_cols] > 0.5

        dist_num_value_pred=self.rank_head(schema_encoding_full[:,:,:num_sel_cols])[...,0].permute(0,2,1)[table_indice]#K,r
        dist_num_value_target=value_target_full.reshape(batch_size, max_row_num, -1).permute(0,2,1)[table_indice]#K,r
        loss_rank=1-spearmanr(dist_num_value_pred,dist_num_value_target,mask=dist_num_value_target != 0)


        table_logits = schema_encoding[:, :num_sel_cols][table_indice]
        table_targets = table_distributions[:, :num_sel_cols][table_indice]
        is_value_slice = table_targets[:, 0] > -5e10
        table_logits = table_logits[is_value_slice]
        table_targets = table_targets[is_value_slice]  # LS,5
        table_predicts = self.distrib_head(table_logits)  # LS,8
        sgn_value = torch.sgn(table_targets[:, :3])  # LS,3
        log_value = torch.log(1e-7 + torch.abs(table_targets[:, :3]))  # LS,3
        other_value = table_targets[:, 3:]  # LS,2
        table_targets = torch.cat((log_value, sgn_value, other_value), axis=-1)  # LS,8
        loss_distrib = ((table_predicts - table_targets) ** 2).mean()+loss_rank
        if math.isnan(loss_distrib.item()):
            loss_distrib = torch.tensor(0.0).to(input_embedding)

        return loss_distill, loss_mlm, loss_distrib

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def mask(self, input_ids, input_embeddings, raw_segment_ids, embeddings_table):
        probability_matrix = torch.full(raw_segment_ids[:, 0].shape, self.mlm_probability).to(input_embeddings)
        text_masked_indices = torch.bernoulli(probability_matrix).bool()
        text_masked_indices = text_masked_indices.unsqueeze(1).expand(-1, raw_segment_ids.size(1), -1).reshape(
            input_ids.size(0), -1)
        segment_ids = raw_segment_ids.reshape(input_ids.size(0), -1)
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability).to(input_embeddings)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids != self.encoder.tokenizer.num_token_id] = False
        masked_indices[segment_ids == 0] = text_masked_indices[segment_ids == 0]
        masked_indices[input_ids == self.encoder.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.encoder.tokenizer.cls_token_id] = False

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8).to(input_embeddings)).bool() & masked_indices

        input_embeddings = torch.where(indices_replaced.unsqueeze(-1),
                                       embeddings_table(
                                           torch.tensor(self.encoder.tokenizer.mask_token_id).to(input_ids)),
                                       input_embeddings)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5).to(input_embeddings)).bool() & masked_indices & ~indices_replaced

        random_words = torch.randint(self.encoder.tokenizer.vocab_size, input_ids.shape).to(input_ids)

        input_embeddings = torch.where(indices_random.unsqueeze(-1), embeddings_table(random_words), input_embeddings)

        return input_embeddings, masked_indices


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
