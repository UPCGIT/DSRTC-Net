import torch
from torch import nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ECAAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')


    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1)).expand_as(x)
        return y


class DAHCA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.eca = ECAAttention(channels, gamma=1, b=2)
        self.max_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(start_dim=2),  # (B, C, 1)
            nn.Conv1d(1, 1, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.LeakyReLU(),
            nn.Sigmoid()
        )
        nn.init.kaiming_normal_(self.eca.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.max_branch[2].weight)

    def forward(self, x):
        eca_att = self.eca(x)  # (B, C, H, W)

        max_att = nn.AdaptiveMaxPool2d(1)(x)
        max_att = max_att.flatten(start_dim=2)
        max_att = max_att.transpose(1, 2)
        max_att = self.max_branch[2](max_att)
        max_att = self.max_branch[3](max_att)
        max_att = self.max_branch[4](max_att)
        max_att = max_att.transpose(1, 2).unsqueeze(-1).expand_as(x)

        fused_att = eca_att + max_att

        return x * fused_att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.LeakyReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch_size,1,lines,cols)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size,1,lines,cols)
        x = torch.cat([avg_out, max_out], dim=1)  # (batch_size,2,lines,cols)
        x = self.conv1(x)  # (batch_size,1,lines,cols)

        return self.sigmoid(x)


class DSRTCNet(nn.Module):
    def __init__(self,num_bands, num_endmembers,lidar_dims):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.lidar_dims = lidar_dims


        self.spectral_encoder = nn.Sequential(
            nn.Conv2d(self.num_bands, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
            nn.Dropout(),

            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Dropout(),

            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(),

        )

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(self.num_bands, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
            nn.Dropout(),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Dropout(),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(),

        )

        self.spatial_encoder_lidar = nn.Sequential(
            nn.Conv2d(self.lidar_dims, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
            nn.Dropout(),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Dropout(),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Dropout(),

        )
        self.channel_attention = DAHCA(channels=num_bands, kernel_size=3)
        self.spatial_attention=SpatialAttention(5)

        self.ConcateUnit=nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32)
        )

        self.fusion_encoder=nn.Sequential(
            nn.Conv2d(64, num_endmembers, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_endmembers, self.num_bands, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x,y):
        x_spectral_attention=self.channel_attention(x)
        x_spatial_attention=self.spatial_attention(x)
        x_spatial=x_spatial_attention*x
        x_spatial=x_spatial+x

        x_spectral=self.spectral_encoder(x_spectral_attention)
        x_spatial=self.spatial_encoder(x_spatial)

        y_spatial=self.spatial_encoder_lidar(y)

        x_spatial_fusion=self.ConcateUnit(torch.cat([x_spatial,y_spatial],dim=1))
        x_spatial_fusion=x_spatial_fusion+x_spatial

        x_fusion=torch.cat([x_spectral,x_spatial_fusion],dim=1)

        abundance=self.fusion_encoder(x_fusion)
        endmembers=self.decoder[0].weight
        reconstructed=self.decoder(abundance)

        return abundance,endmembers,reconstructed
