import torch
import torch.nn as nn

def spearmanr(pred, target, mask):
    import torchsort
    if mask.sum().item()==0:
        return torch.tensor(1.0).to(pred)
    pred = torchsort.soft_rank(pred)[mask]
    target = torchsort.soft_rank(target)[mask]
    pred = pred - pred.mean()
    pred = pred / (pred.norm()+1e-7)
    target = target - target.mean()
    target = target / (target.norm()+1e-7)
    return (pred * target).sum()

class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.gelu=nn.GELU()

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = self.gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)