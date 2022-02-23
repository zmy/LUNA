import torch
import torch.nn as nn


class ValueEmbedding(nn.Module):

    def __init__(self, emb_size, direct_expand):

        super(ValueEmbedding, self).__init__()
        self.emb_size = emb_size
        self.direct_expand = direct_expand
        self.proj = nn.Linear(2, self.emb_size)

    def forward(self, batch_val, batch_sig, batch_exp):
        if self.direct_expand:
            return batch_val.view(-1, 1).expand(-1, self.emb_size)
        else:
            return self.proj(
                torch.cat([batch_sig.view(-1, 1), batch_exp.view(-1, 1)], dim=1)
            )
