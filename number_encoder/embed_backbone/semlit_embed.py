import torch
import torch.nn as nn
from number_tokenizer import NumVocab


class SemLitEmbedding(nn.Module):
    def __init__(self,
                 hidden_size: int = 128,
                 transformer_num_layers: int = 3,
                 transformer_nhead: int = 4,
                 digit_upper_bound: int = 21,
                 out_emb_size: int = 768,
                 dropout_rate: float = 0.1,
                 ):
        super(SemLitEmbedding, self).__init__()

        # TransPos is not directly imported because the I didn't find a good way to
        # overload the forward function

        self.embedding = torch.nn.Embedding(len(NumVocab), hidden_size)
        self.digit_embedding = torch.nn.Embedding(digit_upper_bound, hidden_size)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(hidden_size, nhead=transformer_nhead),
            num_layers=transformer_num_layers
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=transformer_num_layers,
            bidirectional=True,
            dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_size, out_emb_size)
        self.semantic_fc = nn.Linear(out_emb_size, hidden_size)

    def forward(self,
                batch_token_ids,
                batch_digit_mapping,
                batch_token_mapping,
                batch_semantic_embedding
                ):

        # Get literal embedding, digit embedding and semantic embedding
        embs = self.embedding(batch_token_ids)
        digit_embs = self.digit_embedding(batch_digit_mapping)
        token_semantics = batch_semantic_embedding
        # Add an extra row to the semantics so the trailing padding index 0 is mapped to the extra unused row
        extra_entry = torch.zeros(token_semantics.size(0), 1, token_semantics.size(2)).to(token_semantics)
        token_semantics = torch.cat((extra_entry, token_semantics), dim=1)
        indices = batch_token_mapping.unsqueeze(-1).expand(*batch_token_mapping.size(), token_semantics.size(-1))  # Thx Yijia!
        semantics = torch.gather(token_semantics, 1, indices)
        semantic_embs = self.semantic_fc(semantics)

        # Pass embedding through transformer
        embs = embs + digit_embs + semantic_embs
        key_padding_mask = batch_token_ids == NumVocab.PAD  # (N, S)
        embs = torch.transpose(embs, 0, 1)  # (S, N, E)
        encoded = self.encoder(embs, src_key_padding_mask=key_padding_mask)

        # Pass through LSTM
        output, (h_n, c_n) = self.lstm(encoded)
        out_emb = self.fc(torch.mean(h_n, dim=0))
        return out_emb
