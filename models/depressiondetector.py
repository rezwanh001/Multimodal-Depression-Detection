import torch
import torch.nn as nn
from .base import BaseNet
from .hatnet import HATNet
from .gsabottlenecknet import GSABottleneckNet

class DepressionDetector(BaseNet):
    def __init__(self, d=256, l=6, t_downsample=4):
        super().__init__()

        # Initialize feature extractors # for Audio
        self.gsa_bottleneck = GSABottleneckNet(gsa_input=25, gsa_rel_pos_length=10)

        # Initialize feature extractors # for Video
        self.hatnet = HATNet(num_classes=1, dims=[48, 96, 240, 384], head_dim=48, 
                             expansions=[8, 8, 4, 4], grid_sizes=[8, 7, 7, 1], 
                             ds_ratios=[8, 4, 2, 1], depths=[2, 2, 6, 3])

        self.v_downsample = nn.Sequential(
            nn.Conv1d(136, d, kernel_size=16, stride=t_downsample, padding=8),
            nn.BatchNorm1d(d),
        )
        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l, 
        )
        self.a_downsample = nn.Sequential(
            nn.Conv1d(25, d, kernel_size=16, stride=t_downsample, padding=8),
            nn.BatchNorm1d(d),
        )
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l, 
        )
        self.qa_transform = nn.Linear(d, d)
        self.ka_transform = nn.Linear(d, d)
        self.va_transform = nn.Linear(d, d)
        self.qv_transform = nn.Linear(d, d)
        self.kv_transform = nn.Linear(d, d)
        self.vv_transform = nn.Linear(d, d)
        self.cross_av = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )
        self.cross_va = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )
        self.av_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=2*d, nhead=4, dim_feedforward=2*d, batch_first=True
            ),
            num_layers=l, 
        )
        self.z_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2*d, 1)

    def feature_extractor(self, x):
        xa = x[:, :, 136:]
        xv = x[:, :, :136]

        xa = self.a_downsample(xa.transpose(1, 2)).transpose(1, 2)
        xv = self.v_downsample(xv.transpose(1, 2)).transpose(1, 2)
        # ##-------------------------------------------------------------

        # xa = x[:, :, 136:] # Audio # torch.Size([32, ~2259, 25]) 
        # xv = x[:, :, :136] # Video # torch.Size([32, ~2259, 136])
        # # print("xa xv shape: ")
        # # print(xa.shape, xv.shape) # torch.Size([16, 3167, 25]) torch.Size([16, 3167, 136])

        # # GSABottleneckNet: Audio
        # xa, gsa_cls = self.gsa_bottleneck(xa) # xa = torch.Size([32, 256, 71])

        # # HATNET: Video
        # xv = xv.permute(0, 2, 1)
        # xv, hatnet_cls = self.hatnet(xv) # xv = torch.Size([32, 384, 71])
        # # print("gsa, hatnet")
        # # print(xa.shape, xv.shape) # torch.Size([16, 256, 99]) torch.Size([16, 384, 99])

        # # print("audio downsampling")
        # # print(xa.transpose(1, 2).shape) # torch.Size([16, 25, 3167])
        # # print((xa.transpose(1, 2)).transpose(1, 2).shape) # torch.Size([16, 3167, 25])

        # # print("video downsampling")
        # # print(xv.transpose(1, 2).shape) # torch.Size([16, 136, 3167])
        # # print((xv.transpose(1, 2)).transpose(1, 2).shape) # torch.Size([16, 3167, 136])

        # # downsampling 
        # xa = self.a_downsample(xa).transpose(1, 2)
        # xv = self.v_downsample(xv).transpose(1, 2)
        # # print("downsampling: ")
        # # print(xa.shape, xv.shape) # torch.Size([16, 792, 256]) torch.Size([16, 792, 256])

        ##-------------------------------------------------------------
        ua = self.a_encoder(xa) # torch.Size([16, 792, 256])
        uv = self.v_encoder(xv) # torch.Size([16, 792, 256])

        qa = self.qa_transform(ua)
        ka = self.ka_transform(ua)
        va = self.va_transform(ua)
        qv = self.qv_transform(uv)
        kv = self.kv_transform(uv)
        vv = self.vv_transform(uv)

        uua = self.cross_av(qa, kv, vv)[0]
        uuv = self.cross_va(qv, ka, va)[0]
        uav = torch.cat((uua, uuv), dim=2)

        z = self.av_encoder(uav)
        z = torch.mean(z, dim=1)
        # return self.z_dropout(z), gsa_cls, hatnet_cls
        return self.z_dropout(z)
    

    def classifier(self, x):
        z = x
        # z, gsa_cls, hatnet_cls = x
        # return self.fc(z), gsa_cls, hatnet_cls
        return self.fc(z)
