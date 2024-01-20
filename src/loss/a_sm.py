import torch
from torch import nn
import torch.nn.functional as F


class ASoftmax(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, margin: int, eps: float, *args, **kwargs
    ):
        super().__init__()

        self.margin = margin
        self.eps = eps

        self.w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        cos = F.normalize(logits) @ F.normalize(self.w)
        theta = torch.diagonal(cos.transpose(0, 1)[targets])
        theta = torch.clamp(theta, -1 + self.eps, 1 - self.eps)
        num = torch.cos(torch.acos(theta) * self.margin)

        index = torch.ones_like(cos).bool()
        index.scatter_(1, targets.view(-1, 1), False)

        den = cos[index].view(cos.size(0), -1)
        den = torch.exp(num) + torch.sum(torch.exp(den), dim=1)
        den = torch.clamp(den, self.eps)

        return -torch.mean(num - torch.log(den))
