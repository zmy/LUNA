'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
from torch import nn
from transformers import BertForMaskedLM
from .roberta_num import RobertaNum

from number_encoder.numbed import NumBed, NumBedConfig
from .utils import FFNLayer


class BertNum(RobertaNum):
    def build_model(self, config, tokenizer):
        self.encoder = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.encoder.resize_token_embeddings(len(tokenizer))
        # create momentum models
        self.encoder_m = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.encoder_m.resize_token_embeddings(len(tokenizer))
        self.model_pairs = [[self.encoder, self.encoder_m]]
        self.copy_params()
        if self.use_numbed:
            number_model_config = NumBedConfig(model_name=config['numbed_model'], \
                                               encoder_name='TaPas', \
                                               checkpoint_path=config['numbed_ckpt'],
                                               prompt_layers=None if not config['use_prompt'] else 12)
            self.number_model = NumBed(number_model_config)
            self.number_proj = nn.Sequential()
        self.backbone = 'bert'
        self.hidden_size = 768
