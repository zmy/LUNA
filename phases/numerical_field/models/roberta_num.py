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
from transformers import RobertaForMaskedLM

from number_encoder.numbed import NumBed, NumBedConfig
from number_tokenizer.numtok import NumTok
from .utils import spearmanr, FFNLayer


class RobertaNum(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.momentum = config['momentum']
        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        self.use_rank = config['use_rank']
        self.use_logflat = config['use_logflat']
        self.use_distrib = config['use_distrib']
        self.use_numbed = config['numbed_model'] != 'zero'
        self.use_regression = config['use_regression']
        self.loss_ce = nn.CrossEntropyLoss(reduction='sum')
        self.reg_func = nn.SmoothL1Loss(reduction='none') if config['use_huber'] else nn.MSELoss(reduction='none')
        if config['use_firsttoken']:
            self.column_pooling_func = lambda logits, mask: logits[:, 0] if logits.size(1) > 0 else logits.sum(1)
        else:
            self.column_pooling_func = lambda logits, mask: logits.sum(1) / (mask.sum(1, keepdim=True) + 1e-7)
        self.build_model(config, tokenizer)
        if config['use_logflat']:
            self.reg_head = FFNLayer(self.hidden_size, self.hidden_size, 1, 0.1)
            self.distrib_head = FFNLayer(self.hidden_size, self.hidden_size, 5, 0.1)
        else:
            self.reg_head = FFNLayer(self.hidden_size, self.hidden_size, 2, 0.1)
            self.distrib_head = FFNLayer(self.hidden_size, self.hidden_size, 8, 0.1)
        self.rank_head = FFNLayer(self.hidden_size, self.hidden_size, 1, 0.1)
        if config['kept_keys'] == '':
            self.kept_keys = ()
        else:
            self.kept_keys = config['kept_keys'].split(',')

    def build_model(self, config, tokenizer):
        self.encoder = RobertaForMaskedLM.from_pretrained('data/ckpt/roberta.large')
        self.encoder.resize_token_embeddings(len(tokenizer))
        # create momentum models
        self.encoder_m = RobertaForMaskedLM.from_pretrained('data/ckpt/roberta.large')
        self.encoder_m.resize_token_embeddings(len(tokenizer))
        self.model_pairs = [[self.encoder, self.encoder_m]]
        self.copy_params()
        if self.use_numbed:
            number_model_config = NumBedConfig(model_name=config['numbed_model'], \
                                               encoder_name='RoBERTa', \
                                               checkpoint_path=config['numbed_ckpt'])
            self.number_model = NumBed(number_model_config)
            self.number_proj = nn.Sequential()
        self.backbone = 'roberta'
        self.hidden_size = 1024

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                number_list,
                table_distributions,
                column_ids=None,
                token_type_ids=None,
                **kargs
                ):
        """
        for roberta. column_ids is required; while for tapas, column_ids is not required.
        """
        # ===prepare input===
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if column_ids is None:
            column_ids = token_type_ids[..., 1]
            if self.backbone == 'bert':
                token_type_ids = token_type_ids[..., 0]
        word_embedding = getattr(self.encoder, self.backbone).embeddings.word_embeddings(input_ids)
        num_id = torch.zeros_like(input_ids)
        num_indice = input_ids == self.tokenizer.num_token_id
        if len(number_list) > 0:
            num_id[num_indice] = torch.range(1, len(number_list)).to(input_ids)
            if self.use_numbed:
                tokenized = NumTok.tokenize([(x[0], None, None) for x in number_list], input_ids.device, self.kept_keys)
                num_embedding = self.number_model(tokenized)
                num_embedding = self.number_proj(num_embedding)
                num_embedding = torch.cat((torch.zeros((1, self.hidden_size)).to(num_embedding), num_embedding), axis=0)
                num_embedding = F.embedding(num_id, num_embedding)
                input_embedding = word_embedding + num_embedding
            else:
                input_embedding = word_embedding
        else:
            input_embedding = word_embedding
        mask_embedding, masked_indice = self.mask(input_ids, input_embedding,
                                                  getattr(self.encoder, self.backbone).embeddings.word_embeddings)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            mlm_output_m = self.encoder_m(inputs_embeds=mask_embedding,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          return_dict=True)

        mlm_output = self.encoder(inputs_embeds=mask_embedding,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  return_dict=True,
                                  output_hidden_states=True)

        output = getattr(self.encoder, self.backbone)(inputs_embeds=input_embedding,
                                                      attention_mask=attention_mask,
                                                      token_type_ids=token_type_ids,
                                                      return_dict=True)
        # ===distillation loss===

        soft_label = F.softmax(mlm_output_m.logits[masked_indice], dim=-1)  # K,C
        logits = mlm_output.logits[masked_indice]  # K,C
        loss_distill = -torch.sum(F.log_softmax(logits, dim=-1) * soft_label, dim=-1).mean()
        if math.isnan(loss_distill.item()):
            loss_distill = torch.tensor(0.0).to(input_embedding)

        # ===mlm loss===
        value_list = torch.tensor([0] + [x[1] for x in number_list]).to(input_embedding)

        if self.use_logflat:
            value = (torch.sgn(value_list) * torch.log(1 + torch.abs(value_list))).unsqueeze(-1)
        else:
            log_value = torch.log(1e-7 + torch.abs(value_list))
            sgn_value = torch.sgn(value_list)
            value = torch.stack((log_value, sgn_value)).T

        value_target = F.embedding(num_id[num_indice & masked_indice], value)  # K,2
        value_logits = mlm_output.hidden_states[-1][num_indice & masked_indice]  # K,D
        value_pred = self.reg_head(value_logits)  # K,2
        mlm_reg_loss = self.reg_func(value_pred, value_target).mean(-1).sum()
        if math.isnan(mlm_reg_loss.item()) or not self.use_regression:
            mlm_reg_loss = torch.tensor(0.0).to(input_embedding)

        language_target = input_ids[~num_indice & masked_indice]  # K
        language_logits = mlm_output.logits[~num_indice & masked_indice]  # K,C
        mlm_cla_loss = self.loss_ce(language_logits, language_target)
        if math.isnan(mlm_cla_loss.item()):
            mlm_cla_loss = torch.tensor(0.0).to(input_embedding)

        loss_mlm = (mlm_reg_loss + mlm_cla_loss) / (value_target.size(0) + language_target.size(0) + 1e-7)

        # ===distribution===
        if self.use_distrib:
            col_ids = column_ids
            dist_indice = (col_ids <= 50) & (col_ids > 0)
            dist_logits = output.last_hidden_state[dist_indice]  # K,D
            dist_num_id = num_id[dist_indice]  # K
            batch_ids = torch.range(0, input_ids.size(0) - 1).view(-1, 1).to(input_ids)
            flatten_ids = (col_ids - 1) + 50 * batch_ids
            flatten_ids = flatten_ids[dist_indice]  # K
            table_distributions = table_distributions.view(-1, 5)  # 50N,5
            dist_targets = F.embedding(flatten_ids, table_distributions)  # K,5
            is_value_slice = dist_targets[:, 0] > -5e10  # K

            flatten_ids = flatten_ids[is_value_slice].cpu().numpy()  # LK
            dist_logits = dist_logits[is_value_slice]  # LK,D
            dist_num_id = dist_num_id[is_value_slice]  # K
            dist_num_value = F.embedding(dist_num_id, value_list)  # K

            set_flatten_ids = list(set(flatten_ids))
            dict_flatten_ids = {flatten_id: i for i, flatten_id in enumerate(set_flatten_ids)}
            gather_indice = [[] for _ in set_flatten_ids]
            for i, flatten_id in enumerate(flatten_ids):
                if len(gather_indice[dict_flatten_ids[flatten_id]]) < 50:
                    gather_indice[dict_flatten_ids[flatten_id]].append(i)
            max_gather_length = max([len(x) for x in gather_indice] + [0])
            padding_gather_id = len(flatten_ids)
            for x in gather_indice:
                x.extend([padding_gather_id] * (max_gather_length - len(x)))
            if len(gather_indice) != 0:
                gather_indice = torch.tensor(gather_indice).to(input_ids)  # LS,max_gather_length
            else:
                gather_indice = torch.zeros(0, 0).to(input_ids)
            dist_logits = F.embedding(gather_indice,
                                      torch.cat((dist_logits, torch.zeros((1, self.hidden_size)).to(input_embedding)),
                                                axis=0))  # LS,max_gather_length,D

            if self.use_rank:
                dist_num_value_target = F.embedding(gather_indice,
                                                    torch.cat((dist_num_value, torch.zeros(1).to(input_embedding)),
                                                              axis=0))  # LS,max_gather_length
                dist_num_value_pred = self.rank_head(dist_logits)[:, :, 0]  # LS,max_gather_length
                loss_rank = 1 - spearmanr(dist_num_value_pred, dist_num_value_target,
                                          mask=gather_indice != padding_gather_id)
                if math.isnan(loss_rank.item()):
                    loss_rank = torch.tensor(0.0).to(input_embedding)

            dist_logits = self.column_pooling_func(dist_logits, gather_indice != padding_gather_id)  # LS,D
            dist_predicts = self.distrib_head(dist_logits)  # LS,8
            set_flatten_ids = torch.tensor(set_flatten_ids).to(input_ids)  # LS
            dist_targets = F.embedding(set_flatten_ids, table_distributions)  # LS,5

            if self.use_logflat:
                logflat_value = torch.sgn(dist_targets[:, :3]) * torch.log(1 + torch.abs(dist_targets[:, :3]))  # LS,3
                other_value = dist_targets[:, 3:]  # LS,2
                dist_targets = torch.cat((logflat_value, other_value), axis=-1)  # LS,5
            else:
                sgn_value = torch.sgn(dist_targets[:, :3])  # LS,3
                log_value = torch.log(1e-7 + torch.abs(dist_targets[:, :3]))  # LS,3
                other_value = dist_targets[:, 3:]  # LS,2
                dist_targets = torch.cat((log_value, sgn_value, other_value), axis=-1)  # LS,8
            loss_distrib = self.reg_func(dist_predicts, dist_targets).mean()
            if math.isnan(loss_distrib.item()):
                loss_distrib = torch.tensor(0.0).to(input_embedding)
            if self.use_rank:
                loss_distrib = loss_distrib + loss_rank
        else:
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

    def mask(self, input_ids, input_embeddings, embeddings_table):
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability).to(input_embeddings)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8).to(input_embeddings)).bool() & masked_indices

        input_embeddings = torch.where(indices_replaced.unsqueeze(-1),
                                       embeddings_table(torch.tensor(self.tokenizer.mask_token_id).to(input_ids)),
                                       input_embeddings)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5).to(input_embeddings)).bool() & masked_indices & ~indices_replaced

        random_words = torch.randint(self.tokenizer.vocab_size, input_ids.shape).to(input_ids)

        input_embeddings = torch.where(indices_random.unsqueeze(-1), embeddings_table(random_words), input_embeddings)

        return input_embeddings, masked_indices
