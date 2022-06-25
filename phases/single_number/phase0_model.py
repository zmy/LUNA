import torch
import torch.nn as nn
from typing import List, Tuple

from number_encoder.config import NumBedConfig
from number_encoder.numbed import NumBed


class Phase0(nn.Module):
    def __init__(self, config: NumBedConfig):
        super(Phase0, self).__init__()
        self.core_model = NumBed(config)
        self.aligned = config.aligned
        self.align_with_orig = config.align_with_orig

        # Decoding heads
        self.decoding_sig = nn.Sequential(
            nn.Linear(config.out_emb_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )
        self.decoding_exp = nn.Sequential(
            nn.Linear(config.out_emb_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.decoding_exp_class),
            nn.LogSoftmax(dim=-1),
        )
        self.decoding_log = nn.Sequential(
            nn.Linear(config.out_emb_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )

        # Addition heads
        self.addition_sig = nn.Sequential(
            nn.Linear(config.out_emb_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, 1)
        )
        self.addition_exp = nn.Sequential(
            nn.Linear(config.out_emb_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.addition_exp_class),
            nn.LogSoftmax(dim=-1),
        )
        self.addition_log = nn.Sequential(
            nn.Linear(config.out_emb_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, 1),
        )

        # Subtraction heads
        self.subtraction_sig = nn.Sequential(
            nn.Linear(config.out_emb_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, 1)
        )
        self.subtraction_exp = nn.Sequential(
            nn.Linear(config.out_emb_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.addition_exp_class),
            nn.LogSoftmax(dim=-1),
        )
        self.subtraction_log = nn.Sequential(
            nn.Linear(config.out_emb_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, 1),
        )

        # ListMax heads
        self.listmax_bilstm = nn.LSTM(
            input_size=config.out_emb_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.listmax_pred_max_id = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.listmax_class),
            nn.LogSoftmax(dim=-1),
        )

        # Format heads
        self.format_frac_digit = nn.Sequential(
            nn.Linear(config.out_emb_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.format_frac_digit_class),
            nn.LogSoftmax(dim=-1),
        )
        self.format_in01 = nn.Sequential(
            nn.Linear(config.out_emb_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.format_in01_class),
            nn.LogSoftmax(dim=-1),
        )
        self.format_in0100 = nn.Sequential(
            nn.Linear(config.out_emb_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.format_in0100_class),
            nn.LogSoftmax(dim=-1),
        )
        self.format_pred_cp = nn.Sequential(
            nn.Linear(config.out_emb_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, 1)
        )
        self.format_pred_cs = nn.Sequential(
            nn.Linear(config.out_emb_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, 1)
        )

    def forward(self, task, input_dicts):
        if task == 'single':
            inputs = input_dicts[0]
            out = self.core_model(inputs)
            if self.aligned:
                emb, char_feat, num_feat = out
                feats = [(char_feat, num_feat)]
            elif self.align_with_orig:
                emb, numbed_feat, orig_feat = out
                feats = [(numbed_feat, orig_feat)]
            else:
                emb = out
                feats = None
            decoding_sig = self.decoding_sig(emb)
            decoding_exp = self.decoding_exp(emb)
            decoding_log = self.decoding_log(emb)
            format_frac = self.format_frac_digit(emb)
            format_in01 = self.format_in01(emb)
            format_in0100 = self.format_in0100(emb)
            return (decoding_sig, decoding_exp, decoding_log,
                    format_frac, format_in01, format_in0100, feats)

        if task == 'double':
            out = [self.core_model(inputs) for inputs in input_dicts]
            if self.aligned:
                embs = [_[0] for _ in out]
                char_feats = [_[1] for _ in out]
                num_feats = [_[2] for _ in out]
                feats = list(zip(char_feats, num_feats))
            if self.align_with_orig:
                embs = [_[0] for _ in out]
                feats = None
            else:
                embs = out
                feats = None
            emb = torch.cat(embs, dim=1)
            addition_sig = self.addition_sig(emb)
            addition_exp = self.addition_exp(emb)
            addition_log = self.addition_log(emb)
            subtraction_sig = self.subtraction_sig(emb)
            subtraction_exp = self.subtraction_exp(emb)
            subtraction_log = self.subtraction_log(emb)
            cp_pred = self.format_pred_cp(emb)
            cs_pred = self.format_pred_cs(emb)
            return (addition_sig, addition_exp, addition_log,
                    subtraction_sig, subtraction_exp, subtraction_log,
                    cp_pred, cs_pred, feats)

        if task == 'multi':
            out = [self.core_model(inputs) for inputs in input_dicts]
            if self.aligned:
                embs = [_[0] for _ in out]
                char_feats = [_[1] for _ in out]
                num_feats = [_[2] for _ in out]
                feats = list(zip(char_feats, num_feats))
            if self.align_with_orig:
                embs = [_[0] for _ in out]
                feats = None
            else:
                embs = out
                feats = None
            emb = torch.stack(embs).transpose(0, 1)
            _, (last_layers, _) = self.listmax_bilstm(emb)
            last_layers = torch.mean(last_layers, dim=0)
            max_id_pred = self.listmax_pred_max_id(last_layers)
            max_id_pred = max_id_pred.squeeze(0)
            return max_id_pred, feats


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, feats: List[Tuple[torch.Tensor, torch.Tensor]]):
        ''' feats should be tensors of shape [batch_size, out_emb_size]'''
        if feats is None:
            return torch.tensor([0])
        loss = torch.mean(torch.tensor([self.triplet(feat1, feat2) for (feat1, feat2) in feats]))
        return loss

    def triplet(self, feat1, feat2, eps=0.05):
        feat1 = feat1.unsqueeze(1)
        feat2 = feat2.unsqueeze(0)
        feat_num = feat1.size(0)
        similarity = nn.CosineSimilarity(dim=2, eps=1e-6)(feat1, feat2)
        # similarity = torch.mm(feat1, feat2.t())
        pos_ind = [[i] for i in range(feat_num)]
        neg_ind = [[j for j in range(feat_num) if j != i] for i in range(feat_num)]
        pos_ind = torch.tensor(pos_ind, dtype=torch.long).to(feat1.device)
        neg_ind = torch.tensor(neg_ind, dtype=torch.long).to(feat2.device)
        pos = torch.gather(similarity, 1, pos_ind)
        neg = torch.gather(similarity, 1, neg_ind)
        triplet_loss = torch.mean(torch.clamp(neg-pos+eps, min=0))
        return triplet_loss
