# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from models.utils import get_backbone_model, ProjectionHead


class BarlowTwins(nn.Module):
    def __init__(self, backbone_name, backbone_parameter, projection_hidden, projection_size, lambd=0.005):
        super().__init__()
        self.backbone = get_backbone_model(model_name=backbone_name,
                                           parameters=backbone_parameter)
        self.projection = ProjectionHead(in_features=self.backbone.final_length,
                                         hidden_features=projection_hidden,
                                         out_features=projection_size)
        self.bn = nn.BatchNorm1d(projection_size, affine=False)
        self.lambd = lambd

    def forward(self, x):
        x1, x2 = x
        batch_size = x[0].shape[0]
        z1 = self.projection(self.backbone(x1))
        z2 = self.projection(self.backbone(x2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
