# This code is from https://github.com/wjdghks950/Methods-for-Numeracy-Preserving-Word-Embeddings/blob/master/dice_embedding.ipynb

import torch
import torch.nn as nn
import math


class DiceEmbedding(nn.Module):

    def __init__(self,
                 emb_size=1024,
                 mode='log',
                 d=10,
                 norm="l2",
                 dice_max=100000,
                 dice_min=-100000,
                 dice_log_min=-5,
                 dice_log_max=80):
        super(DiceEmbedding, self).__init__()
        self.d = d  # By default, we build DICE-2
        self.mode = mode
        if mode == 'val':
            self.min_bound = -100000
            self.max_bound = 100000
        elif mode == 'log':
            self.min_bound = -5
            self.max_bound = 80
        self.norm = norm  # Restrict x and y to be of unit length
        self.M = torch.normal(0, 1, (self.d, self.d))
        self.Q, self.R = torch.qr(self.M, some=False)  # QR decomposition for orthonormal basis, Q
        self.proj = nn.Linear(d, emb_size)
        self.dice_min = dice_min
        self.dice_max = dice_max

    def __linear_mapping(self, num):
        norm_diff = num / abs(self.min_bound - self.max_bound)
        theta = norm_diff * math.pi
        return theta

    def dice_embed(self, num):
        theta = self.__linear_mapping(num)
        if self.d == 2:
            # DICE-2
            polar_coord = torch.tensor([math.cos(theta), math.sin(theta)])
        elif self.d > 2:
            # DICE-D
            polar_coord = torch.tensor([math.sin(theta)**(dim-1) * math.cos(theta) if dim < self.d
                                        else math.sin(theta)**(self.d)
                                        for dim in range(1, self.d+1)])
        else:
            raise ValueError("Wrong value for `d`. `d` should be greater than or equal to 2.")

        dice = torch.sum(self.Q * polar_coord, dim=-1)
        return dice

    def forward(self, batch_val):
        if self.mode == 'val':
            batch_val = torch.clamp(batch_val, min=self.dice_min, max=self.dice_max)
            dice_emb = torch.stack([self.dice_embed(val) for val in batch_val]).to(batch_val.device)
        elif self.mode == 'log':
            batch_log = torch.log(0.01 + torch.abs(batch_val))
            dice_emb = torch.stack([self.dice_embed(val) for val in batch_log]).to(batch_val.device)
        return self.proj(dice_emb)
