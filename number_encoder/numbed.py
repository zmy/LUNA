from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_states, num_params
from .config import NumBedConfig
from .embed_backbone import CharLSTM, Hybrid, RoBERTaEmbedding, ValueEmbedding, DiceEmbedding, TutaFeatEmbedding
from .embed_backbone import SemLitEmbedding, TransPosEmbedding, StrealEmbedding
from transformers import RobertaModel


class NumBed(nn.Module):
    def __init__(self, config: NumBedConfig):
        super(NumBed, self).__init__()
        assert config.model_name is not None
        if config.prompt_layers is not None:
            assert config.model_name in {'CharLSTM','TransPos'}
        self.allow_non_param_keys=()
        if config.model_name == 'CharLSTM':
            self.model = CharLSTM(
                model_id=config.get_model_id(),
                out_emb_size=config.out_emb_size,
                hidden_size=config.hidden_size,
                lstm_num_layers=config.lstm_num_layers,
                bidirectional=config.bidirectional,
                preprocess_type=config.preprocess_type,
                prompt_layers=config.prompt_layers,
            )
            print('Built CharLSTM model! Number of parameters: ', num_params(self.model))
            self.param_keys = ('batch_token_ids', 'batch_seq_len')
            self.allow_non_param_keys = ('batch_pos_embed',)

        elif config.model_name == 'Hybrid':
            self.model = Hybrid(
                model_id=config.get_model_id(),
                emb_size=config.emb_size,
                lstm_num_layers=config.lstm_num_layers,
                bidirectional=config.bidirectional,
                preprocess_type=config.preprocess_type,
                value_ratio=config.value_ratio,
                mix=config.mix,
                aligned=config.aligned,
            )
            print('Built Hybrid model! Number of parameters: ', num_params(self.model))
            self.param_keys = ('batch_token_ids', 'batch_seq_len',
                               'batch_sig', 'batch_exp')

        elif config.model_name == 'RoBERTa':
            self.model = RoBERTaEmbedding()
            self.param_keys = ('input_ids', 'attention_mask')

        elif config.model_name == 'ValueEmbedding':
            self.model = ValueEmbedding(emb_size=config.emb_size,
                                        direct_expand=config.direct_expand)
            print('Built ValueEmbedding model! Number of parameters: ', num_params(self.model))
            self.param_keys = ('batch_val', 'batch_sig', 'batch_exp')

        elif config.model_name == 'Dice':
            self.model = DiceEmbedding(emb_size=config.emb_size, mode=config.mode)
            print('Built DICE model! Number of parameters: ', num_params(self.model))
            self.param_keys = ('batch_val',)

        elif config.model_name == 'TutaFeat':
            self.model = TutaFeatEmbedding(out_emb_size=config.out_emb_size)
            print('Built TutaFeat model! Number of parameters: ', num_params(self.model))
            self.param_keys = ('batch_tuta_feat',)

        elif config.model_name == 'TransPos':
            self.model = TransPosEmbedding(hidden_size=config.hidden_size,
                                           transformer_num_layers=config.transformer_num_layers,
                                           transformer_nhead=config.transformer_nhead,
                                           direct_average=config.direct_average,
                                           out_emb_size=config.out_emb_size,
                                           prompt_layers=config.prompt_layers,)
            print('Built TransPos model! Number of parameters: ', num_params(self.model))
            self.param_keys = ('batch_token_ids', 'batch_digit_mapping')
            self.allow_non_param_keys = ('batch_pos_embed',)

        elif config.model_name == 'Streal':
            self.model = StrealEmbedding(hidden_size=config.hidden_size,
                                         transformer_num_layers=config.transformer_num_layers,
                                         transformer_nhead=config.transformer_nhead,
                                         out_emb_size=config.out_emb_size)
            print('Built Streal model! Number of parameters: ', num_params(self.model))
            self.param_keys = ('batch_token_ids',
                               'batch_digit_mapping',
                               'batch_tuta_feat',
                               'batch_sig',
                               'batch_exp',
                               'batch_format_feat')

        elif config.model_name == 'SemLit':
            self.model = SemLitEmbedding(hidden_size=config.hidden_size,
                                         transformer_num_layers=config.transformer_num_layers,
                                         transformer_nhead=config.transformer_nhead,
                                         out_emb_size=config.out_emb_size)
            print('Built SemLit model! Number of parameters: ', num_params(self.model))
            self.param_keys = ('batch_token_ids',
                               'batch_digit_mapping',
                               'batch_token_mapping',
                               'batch_semantic_embedding')
        else:
            raise ValueError(f"Unexpected model name: '{config.model_name}'")

        if config.checkpoint_path != '':
            print('loading from ', config.checkpoint_path)
            print(self.model.load_state_dict(load_states(config.checkpoint_path)))
        self.model_id = config.get_model_id()
        self.emb_size = config.emb_size
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.aligned = config.aligned
        self.use_layer_norm = config.use_layer_norm
        self.align_with_orig = config.align_with_orig
        if self.align_with_orig:
            if self.emb_size == 768:
                self.orig_model = RobertaModel.from_pretrained('roberta-base')
            elif self.emb_size == 1024:
                self.orig_model = RobertaModel.from_pretrained('roberta-large')
            else:
                raise ValueError('align_with_orig encountered with an unknown dimension')
        if self.use_layer_norm:
            self.layernorm = nn.LayerNorm(config.emb_size)
        self.model_name = config.model_name

    def forward(self, input_dict: Dict) -> torch.Tensor:
        # pass in corresponding parameters from input dict to produce embs.
        if len(input_dict) == 0:
            return torch.zeros(0, self.emb_size, device=self.dummy_param.device)
        new_input_dict={key: input_dict[key] for key in self.param_keys}
        new_input_dict.update({key: input_dict.get(key,None) for key in self.allow_non_param_keys})
        out = self.model(**new_input_dict)

        # Posprocessing to the backbone output
        if self.aligned:
            # self.model is Hybrid, model returns embs, last_layers_projected, sigexp_projected
            # The last two outputs are meant for alignment outside the forward function
            embs, last_layers_projected, sigexp_projected = out
            return embs, last_layers_projected, sigexp_projected

        elif self.align_with_orig:
            # self.model is CharLSTM, model returns embs, embs, original_embs
            # The last two outputs are meant for alignment outside the forward function
            embs = out
            if self.use_layer_norm:
                embs = self.layernorm(embs)
            orig_embs = self.get_orig_emb(input_dict['input_ids'], input_dict['attention_mask'])
            return embs, embs, orig_embs

        else:
            # Other models, model returns embs
            embs = out
            if len(embs.shape) == 1:
                embs = embs.view(1, -1)
            return self.layernorm(embs) if self.use_layer_norm else embs

    def prompting(self, input_ids: torch.LongTensor, numtok_dict: Dict, num_token_id: int) -> Optional[torch.Tensor]:
        if len(numtok_dict) == 0:
            return None

        num_cnt = numtok_dict['batch_token_ids'].shape[0]
        num_embed = self(numtok_dict)
        sample_num=(input_ids==num_token_id).sum(1).cpu().numpy()
        sample_max=sample_num.max()
        import numpy as np
        prompt=np.full((sample_num.shape[0],sample_max),sample_num.sum(),dtype=np.int64)
        of_set=0
        for i in range(sample_num.shape[0]):
            prompt[i,:sample_num[i]]=list(range(of_set,of_set+sample_num[i]))
            of_set+=sample_num[i]
        prompt_mask=torch.from_numpy(prompt!=sample_num.sum()).to(input_ids)
        prompt=torch.from_numpy(prompt).to(input_ids)
        num_embed = torch.cat((num_embed,torch.zeros(1, *(num_embed.size()[1:])).to(num_embed)), dim=0)
        prompt = F.embedding(prompt, num_embed)
        return prompt,prompt_mask

    def embedding(self, input_ids: torch.LongTensor, numtok_dict: Dict, num_token_id: int) -> Optional[torch.Tensor]:
        if len(numtok_dict) == 0:
            return None
        num_cnt = numtok_dict['batch_token_ids'].shape[0]
        num_embed = self(numtok_dict)
        embed_size = num_embed.shape[1]
        num_embed = torch.cat((torch.zeros(1, embed_size).to(num_embed), num_embed), axis=0)
        num_index = torch.zeros_like(input_ids)
        num_index[input_ids == num_token_id] = torch.range(1, num_cnt).to(input_ids)
        num_embed = F.embedding(num_index, num_embed)
        return num_embed

    def get_orig_emb(self, input_ids, attention_mask):
        return self.get_semantic_emb(self.orig_model, input_ids, attention_mask)

    @staticmethod
    def get_semantic_emb(encoder, input_ids, attention_mask):
        orig_output = encoder(input_ids, attention_mask)
        emb = orig_output['last_hidden_state']

        # get rid of <s> and </s>
        effective_length = torch.sum(attention_mask, dim=1)
        attention_mask[torch.arange(attention_mask.shape[0]), effective_length-1] = 0
        attention_mask[:, 0] = 0

        # get average
        emb = emb * attention_mask.unsqueeze(-1)
        result = torch.sum(emb, dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)

        # result of shape (batch_size, emb_size)
        return result
