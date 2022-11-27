import torch.nn as nn


class TorchModel_v0(nn.Module):
    def __init__(self,window):

        super().__init__()
        self.out_ch_0 = 8
        self.seq_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.out_ch_0,
                kernel_size=3,
              #  stride=(1, 1),
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_0),
        )
        self.out_ch_1 = 8
        self.seq_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_ch_0,
                out_channels=self.out_ch_1,
                kernel_size=3,
             #   stride=(2, 1),
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_1),
        )
        self.out_ch_2 = 8
        self.seq_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_ch_1,
                out_channels=self.out_ch_2,
                kernel_size=3,
               # stride=(2, 1),
                padding=1,
                padding_mode="reflect",
            ),            
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_2),
        )
        self.out_ch_3 = 8
        self.seq_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_ch_2,
                out_channels=self.out_ch_3,
                kernel_size=3,
                #stride=(2, 1),
                padding=1,                        
            ),            
            nn.ReLU(),
            nn.BatchNorm2d(self.out_ch_3),
            nn.AvgPool3d(kernel_size=(8,11,1)),            
        )

        self.n_hidden = 15
        self.linear_0 = nn.Sequential(
            nn.Linear(self.n_hidden, 1),
        )

    def forward(self, x):        
        x = self.seq_0(x)
        x = self.seq_1(x)
        x = self.seq_2(x)
        x = self.seq_3(x)
        x=x.squeeze()        
        
        x = self.linear_0(x)
        return x.squeeze()

    