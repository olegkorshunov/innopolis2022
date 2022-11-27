import torch.nn as nn


class TorchModel(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.n_hidden = 32
        self.k_size = 3
        self.pad = 1
        self.cnn_1 = nn.Sequential(
            # 1
            nn.Conv1d(in_channels=1, out_channels=self.n_hidden,
                      kernel_size=self.k_size, padding=self.pad, padding_mode='reflect'),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Dropout1d(p=0.60),
            nn.ReLU(),
            # 2
            nn.Conv1d(in_channels=self.n_hidden, out_channels=self.n_hidden,
                      kernel_size=self.k_size, padding=self.pad, padding_mode='reflect'),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Dropout1d(p=0.55),
            nn.ReLU(),
            # 3
            nn.Conv1d(in_channels=self.n_hidden, out_channels=16,
                      kernel_size=self.k_size, padding=self.pad, padding_mode='reflect'),
            # nn.BatchNorm1d(16),
            nn.Dropout1d(p=0.3),
            nn.ReLU(),
            # 4
            nn.Conv1d(in_channels=16, out_channels=12,
                      kernel_size=self.k_size, padding=self.pad, padding_mode='reflect'),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Dropout1d(p=0.3),
            nn.ReLU(),
            # 5
            nn.Conv1d(in_channels=12, out_channels=12,
                      kernel_size=self.k_size, padding=self.pad, padding_mode='reflect'),
            nn.BatchNorm1d(12),
            # nn.Dropout1d(p=self.droput),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(12, 1)),
        )

        self.linear = nn.Sequential(
            nn.Linear(70, 7),
        )

    def forward(self, x):
        x = self.cnn_1(x)
        x = x.squeeze()
        x = self.linear(x)
        return x
