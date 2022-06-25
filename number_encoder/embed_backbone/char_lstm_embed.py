import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from number_tokenizer import NUM_PROCESS_FUNCS, NumVocab


class CharLSTM(torch.nn.Module):
    def __init__(self,
                 model_id: str = 'CharLSTM',
                 out_emb_size: int = 768,
                 hidden_size: int = 128,
                 lstm_num_layers: int = 3,
                 bidirectional: bool = True,
                 preprocess_type: str = 'trivial',
                 prompt_layers=None,
                 ):
        super(CharLSTM, self).__init__()
        self.model_id = model_id

        self.embedding = torch.nn.Embedding(len(NumVocab), hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.preprocess = NUM_PROCESS_FUNCS[preprocess_type]
        self.prompt_layers=prompt_layers
        self.out_emb_size=out_emb_size
        if self.prompt_layers is not None:
            self.proj = nn.Linear(hidden_size, out_emb_size*self.prompt_layers*2)
            self.pos_proj = nn.Linear(out_emb_size, hidden_size)
        else:
            self.proj = nn.Linear(hidden_size, out_emb_size)

    def forward(self, batch_token_ids, batch_seq_len, batch_pos_embed=None) -> torch.Tensor:
        batch_token_ids = self.embedding(batch_token_ids)
        packed_inputs = pack_padded_sequence(batch_token_ids,
                                             batch_seq_len.cpu(),
                                             batch_first=True,
                                             enforce_sorted=False)
        _, (last_layers, _) = self.lstm(packed_inputs)
        last_layers = torch.mean(last_layers, dim=0)
        last_layers = last_layers.squeeze(0)
        if self.prompt_layers is None:
            last_layers = self.proj(last_layers)
            return last_layers
        assert batch_pos_embed is not None
        last_layers = self.pos_proj(batch_pos_embed.detach())+last_layers
        last_layers = self.proj(last_layers)#.view(-1,self.prompt_layers,2,self.out_emb_size)
        return last_layers
