import torch
import torch.nn as nn
from number_tokenizer import NumVocab
from utils import num_params


class TransPosEmbedding(nn.Module):
    def __init__(self,
                 hidden_size,
                 transformer_num_layers=3,
                 transformer_nhead=4,
                 direct_average=False,
                 digit_upper_bound=21,
                 out_emb_size=768,
                 dropout_rate=0.1):
        super(TransPosEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(len(NumVocab), hidden_size)
        self.digit_embedding = torch.nn.Embedding(digit_upper_bound, hidden_size)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(hidden_size, nhead=transformer_nhead),
            num_layers=transformer_num_layers
        )
        self.fc = nn.Linear(hidden_size, out_emb_size)
        self.direct_average = direct_average
        if not direct_average:
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=transformer_num_layers,
                bidirectional=True,
                dropout=dropout_rate
            )

    def forward(self, batch_token_ids, batch_digit_mapping):
        embs = self.embedding(batch_token_ids)
        digit_embs = self.digit_embedding(batch_digit_mapping)
        embs += digit_embs
        key_padding_mask = batch_token_ids == NumVocab.PAD  # (N, S)
        embs = torch.transpose(embs, 0, 1)  # (S, N, E)
        encoded = self.encoder(embs, src_key_padding_mask=key_padding_mask)

        if self.direct_average:
            out_emb = self.fc(torch.mean(encoded, dim=0))
        else:
            output, (h_n, c_n) = self.lstm(encoded)
            out_emb = self.fc(torch.mean(h_n, dim=0))
        return out_emb
