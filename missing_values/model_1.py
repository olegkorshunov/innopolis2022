import torch.nn as nn


class TorchModel_v1(nn.Module):
    def __init__(self, window):

        super().__init__()
        self.out_ch_0 = 1
        self.seq_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.out_ch_0,
                kernel_size=3,
                stride=(1, 1),
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_0),
        )
        self.out_ch_1 = 1
        self.seq_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_ch_0,
                out_channels=self.out_ch_1,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_1),
        )
        self.out_ch_2 = 1
        self.seq_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_ch_1,
                out_channels=self.out_ch_2,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_2),
        )
        self.out_ch_3 = 1
        self.seq_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_ch_2,
                out_channels=self.out_ch_3,
                kernel_size=2,
                stride=(1, 1),
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_3),
        )
        self.out_ch_4 = 1
        self.seq_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_ch_3,
                out_channels=self.out_ch_4,
                kernel_size=4,
                stride=(3, 1),
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_4),
        )

        self.n_hidden = window
        self.linear_0 = nn.Sequential(
            nn.Linear(self.n_hidden, 1),
        )

    def forward(self, x):
        x = self.seq_0(x)
        x = self.seq_1(x)
        x = self.seq_2(x)
        x = self.seq_3(x)
        x = self.seq_4(x)

        x = self.linear_0(x)
        return x.squeeze()
