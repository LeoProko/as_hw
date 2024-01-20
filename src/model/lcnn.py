import torch
from torch import nn


class MFMBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        splited = torch.split(x, x.size(-1) // 2, -1)
        return torch.max(splited[0], splited[1])


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
                MFMBlock(),
                MyBatchNorm2d(in_features),
                MyConv2d(
                    in_features, out_features * 2, kernel_size=3, stride=1, padding=1
                ),
                MFMBlock(),
                MyConv2d(
                    out_features, out_features, kernel_size=1, stride=1, padding=0
                ),
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
                MFMBlock(),
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
                nn.Dropout(0.70),
                nn.Linear(9600, 160),
                MFMBlock(),
                nn.BatchNorm1d(80),
                nn.Linear(80, 2),
            ]
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, MyConv2d):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, spectrogram: torch.Tensor, *args, **kwargs):
        # (B, 80, 251): Mel
        # (B, 40, 321): LFCC

        x = self.spec_transform(spectrogram.unsqueeze(-1))
        x = x.flatten(1, 3)
        x = self.pred(x)

        return x
