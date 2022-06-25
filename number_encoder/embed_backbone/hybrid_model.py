import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Tuple

from number_tokenizer import NUM_PROCESS_FUNCS, NumVocab


class Hybrid(torch.nn.Module):
    def __init__(self,
                 model_id,
                 emb_size,
                 lstm_num_layers,
                 bidirectional,
                 preprocess_type,
                 value_ratio=0.25,
                 mix='cat',
                 aligned=False,
                 ):
        super(Hybrid, self).__init__()
        self.model_id = model_id
        self.mix = mix
        self.embedding = torch.nn.Embedding(len(NumVocab), emb_size)
        if mix == 'cat':
            value_dimension_num = int(emb_size * value_ratio)
            lstm_dimension_num = emb_size - value_dimension_num
        elif mix == 'add':
            value_dimension_num = emb_size
            lstm_dimension_num = emb_size
        elif mix == 'cat_proj':
            value_dimension_num = emb_size
            lstm_dimension_num = emb_size

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=lstm_dimension_num,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.aligned = aligned

        self.proj2 = nn.Sequential(
            nn.Linear(2, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, value_dimension_num)
        )
        self.cat_proj = nn.Sequential(
            nn.Linear(2 * emb_size, 2 * emb_size),
            nn.ReLU(),
            nn.Linear(2 * emb_size, emb_size)
        )

        self.preprocess = NUM_PROCESS_FUNCS[preprocess_type]

    def forward(self, batch_token_ids,
                batch_seq_len, batch_sig, batch_exp) -> torch.Tensor:
        batch_token_ids = self.embedding(batch_token_ids)
        packed_inputs = pack_padded_sequence(batch_token_ids,
                                             batch_seq_len.cpu(),
                                             batch_first=True,
                                             enforce_sorted=False)
        _, (last_layers, _) = self.lstm(packed_inputs)
        last_layers = torch.mean(last_layers, dim=0)
        last_layers = last_layers.squeeze(0)
        sigexp = torch.cat((batch_sig.view(-1, 1),
                            batch_exp.float().view(-1, 1)), dim=1)

        last_layers_projected = last_layers
        sigexp_projected = self.proj2(sigexp)
        if self.mix == 'cat':
            out = torch.cat((last_layers_projected, sigexp_projected), dim=1)
        elif self.mix == 'add':
            out = last_layers_projected + sigexp_projected
        elif self.mix == 'cat_proj':
            out = self.cat_proj(torch.cat((last_layers_projected, sigexp_projected), dim=1))

        if self.aligned:
            return out, last_layers_projected, sigexp_projected
        else:
            return out
