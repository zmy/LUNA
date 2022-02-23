import torch
import torch.nn as nn
from .trans_pos import TransPosEmbedding


class StrealEmbedding(nn.Module):
    def __init__(self,
                 hidden_size: int = 32,
                 transformer_num_layers: int = 3,
                 transformer_nhead: int = 4,
                 digit_upper_bound: int = 21,
                 out_emb_size: int = 768,
                 numerical_range: int = 10,
                 numerical_num: int = 4,
                 format_length_upper_bound: int = 13,
                 format_flag_num: int = 2,
                 format_num: int = 4,
                 dropout_rate: float = 0.1,
                 ):
        super(StrealEmbedding, self).__init__()

        # Literal
        self.transpos_embed = TransPosEmbedding(hidden_size=hidden_size,
                                                transformer_num_layers=transformer_num_layers,
                                                transformer_nhead=transformer_nhead,
                                                direct_average=False,
                                                digit_upper_bound=digit_upper_bound,
                                                out_emb_size=out_emb_size,
                                                dropout_rate=dropout_rate)

        # Numerical
        self.mag_embedding = nn.Embedding(numerical_range, out_emb_size // numerical_num)
        self.prec_embedding = nn.Embedding(numerical_range, out_emb_size // numerical_num)
        self.msd_embedding = nn.Embedding(numerical_range, out_emb_size // numerical_num)
        self.lsd_embedding = nn.Embedding(numerical_range, out_emb_size // numerical_num)
        self.sigexp_proj = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2, hidden_size),  # 2 because sigexp consists of sig and exp
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_emb_size)
        )

        # Semantic

        # Format
        self.length_embedding = nn.Embedding(format_length_upper_bound, out_emb_size // format_num)
        self.int_embedding = nn.Embedding(format_flag_num, out_emb_size // format_num)
        self.negative_embedding = nn.Embedding(format_flag_num, out_emb_size // format_num)
        self.percentage_embedding = nn.Embedding(format_flag_num, out_emb_size // format_num)

    def forward(self,
                batch_token_ids,
                batch_digit_mapping,
                batch_tuta_feat,
                batch_sig,
                batch_exp,
                batch_format_feat):
        # Literal
        lit_emb = self.transpos_embed(batch_token_ids, batch_digit_mapping)

        # Numerical
        mag_emb = self.mag_embedding(batch_tuta_feat[:, 0])
        prec_emb = self.prec_embedding(batch_tuta_feat[:, 1])
        msd_emb = self.msd_embedding(batch_tuta_feat[:, 2])
        lsd_emb = self.lsd_embedding(batch_tuta_feat[:, 3])
        numerical_emb = torch.cat((mag_emb, prec_emb, msd_emb, lsd_emb), dim=1)
        sigexp = torch.cat((batch_sig.view(-1, 1), batch_exp.to(batch_sig).view(-1, 1)), dim=1)
        sigexp_projected = self.sigexp_proj(sigexp)
        numerical_emb += sigexp_projected

        # Semantic

        # Format
        batch_length, batch_int, batch_negative, batch_percentage = batch_format_feat
        length_emb = self.length_embedding(batch_length)
        int_emb = self.int_embedding(batch_int)
        negative_emb = self.negative_embedding(batch_negative)
        percentage_emb = self.percentage_embedding(batch_percentage)
        format_emb = torch.cat((length_emb, int_emb, negative_emb, percentage_emb), dim=1)

        return lit_emb + numerical_emb + format_emb
