import torch
from torch import nn


class Transpose(nn.Module):
    def __init__(self, dim_1: int, dim_2: int):
        super().__init__()

        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x: torch.Tensor):
        return x.transpose(self.dim_1, self.dim_2)


class MFM2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # x: (B, H, W, C)

        assert x.size(-1) % 2 == 0

        B, H, W, C = x.shape
        res = torch.zeros(B, H, W, C // 2, device=x.device)

        for i in range(C // 2):
            l = x[:, :, :, i]
            r = x[:, :, :, i + C // 2]

            o = l
            o[l < r] = r[l < r]

            res[:, :, :, i] = o

        return res


class MFM1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # x: (B, C)

        assert x.size(-1) % 2 == 0

        B, C = x.shape

        l = x[:, : C // 2]
        r = x[:, C // 2 :]

        res = l
        res[l < r] = r[l < r]

        return res


class MyConv1d(nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.Conv1d.forward(self, input.transpose(-1, -2)).transpose(-1, -2)


class MyConv2d(nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.Conv2d.forward(self, input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MyMaxPool2d(nn.MaxPool2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.MaxPool2d.forward(self, input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MyBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.BatchNorm2d.forward(self, input.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )


class LCNNBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            *[
                MyConv2d(
                    in_features, in_features * 2, kernel_size=1, stride=1, padding=0
                ),
                MFM2D(),
                MyBatchNorm2d(in_features),
                MyConv2d(
                    in_features, out_features * 2, kernel_size=3, stride=1, padding=1
                ),
                MFM2D(),
                # nn.Dropout(0.1),
                # MyConv2d(
                #     out_features, out_features, kernel_size=1, stride=1, padding=0
                # ),
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class LCNN(nn.Module):
    def __init__(
        self,
        channels: list[int],
    ):
        super().__init__()

        assert len(channels) == 6

        self.spec_transform = nn.Sequential(
            *[
                MyConv2d(1, channels[0], kernel_size=5, stride=1, padding=2),
                MFM2D(),
                MyMaxPool2d(2, 2),
                LCNNBlock(channels[1], channels[2]),
                MyMaxPool2d(2, 2),
                MyBatchNorm2d(channels[2]),
                LCNNBlock(channels[2], channels[3]),
                MyMaxPool2d(2, 2),
                LCNNBlock(channels[3], channels[4]),
                MyBatchNorm2d(channels[4]),
                LCNNBlock(channels[4], channels[5]),
                MyMaxPool2d(2, 2),
            ]
        )
        self.pred = nn.Sequential(
            *[
                nn.Linear(3200, 160),
                MFM1D(),
                nn.BatchNorm1d(80),
                nn.Linear(80, 2),
            ]
        )

    def forward(self, spectrogram: torch.Tensor, *args, **kwargs):
        # (B, 80, 251): Mel
        # (B, 40, 321): LFCC

        x = self.spec_transform(spectrogram.unsqueeze(-1))
        x = x.flatten(1, 3)
        x = self.pred(x)

        return x
