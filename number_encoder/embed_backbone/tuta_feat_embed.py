import torch
import torch.nn as nn

TUTA_FEAT_RANGE = 10  # TUTA features assign one embedding to each digit 0-9
TUTA_FEAT_NUM = 4  # All TUTA features are concatenated to form the final embedding


class TutaFeatEmbedding(nn.Module):

    def __init__(self,
                 out_emb_size: int,
                 tuta_feat_range: int = 10,
                 tuta_feat_num: int = 4,
                 dropout_rate: float = 0.1,
                 hidden_size: int = 768,
                 ):

        super(TutaFeatEmbedding, self).__init__()
        self.mag_embedding = nn.Embedding(tuta_feat_range, out_emb_size // tuta_feat_num)
        self.prec_embedding = nn.Embedding(tuta_feat_range, out_emb_size // tuta_feat_num)
        self.msd_embedding = nn.Embedding(tuta_feat_range, out_emb_size // tuta_feat_num)
        self.lsd_embedding = nn.Embedding(tuta_feat_range, out_emb_size // tuta_feat_num)

        self.mlp = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(out_emb_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_emb_size),
        )

    def forward(self, batch_tuta_feat):
        mags = batch_tuta_feat[:, 0]
        precs = batch_tuta_feat[:, 1]
        msds = batch_tuta_feat[:, 2]
        lsds = batch_tuta_feat[:, 3]

        mag_emb = self.mag_embedding(mags)
        prec_emb = self.prec_embedding(precs)
        msd_emb = self.msd_embedding(msds)
        lsd_emb = self.lsd_embedding(lsds)

        embs = torch.cat((mag_emb, prec_emb, msd_emb, lsd_emb), dim=1)
        embs = self.mlp(embs)
        return embs
