# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------
import torch
import torch.nn as nn
from .base import BaseNet
from .hatnet import HATNet
from .gsabottlenecknet import GSABottleneckNet
# from .mutualtransformer import MutualTransformer

class DepDetAblation(BaseNet):
    def __init__(self, d=256, l=6, ablation='add'):
        """
        Initializes the DepDet model, which combines audio and visual features 
        through various networks and transformers for binary classification.

        Args:
            d (int)             :   Dimensionality for transformer layers and linear layers.
            l (int)             :   Number of layers for the transformers.
            t_downsample (int)  :   Target downsampling size for the audio and video features.
        """
        super().__init__()

        self.ablation=ablation

        # Initialize feature extractors # for Audio
        self.gsa_bottleneck = GSABottleneckNet(gsa_input=25, gsa_rel_pos_length=10)

        # Initialize feature extractors # for Video
        self.hatnet = HATNet(num_classes=1, dims=[48, 96, 240, 384], head_dim=48, 
                             expansions=[8, 8, 4, 4], grid_sizes=[8, 7, 7, 1], 
                             ds_ratios=[8, 4, 2, 1], depths=[2, 2, 6, 3])
        
        # # Initialize mutual transformer for crossing and fusing: video & audio
        # self.mutualtransformer = MutualTransformer(d=d, l=l)

        # Define downsampling layers: audio
        self.audio_downsample = nn.Sequential(
            nn.Conv1d(256, d, kernel_size=1),
            nn.BatchNorm1d(d),
            # nn.AdaptiveAvgPool1d(t_downsample)  # Downsample to the desired length
        )

        # Define downsampling layers: video
        self.video_downsample = nn.Sequential(
            nn.Conv1d(384, d, kernel_size=1),
            nn.BatchNorm1d(d),
            # nn.AdaptiveAvgPool1d(t_downsample)  # Downsample to the desired length
        )

        # Encoder for fused audio-visual features
        self.av_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=8, dim_feedforward=6*d, batch_first=True
            ),
            num_layers=l,
        )

        # Encoder for fused audio-visual features for concatation
        self.av_encoder_concat = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=2*d, nhead=8, dim_feedforward=6*d, batch_first=True
            ),
            num_layers=l,
        )

        if self.ablation == 'add' or self.ablation == 'multi':
            # classification layer
            self.z_dropout = nn.Dropout(0.35)
            self.fc = nn.Linear(d, 1)
        elif self.ablation == 'concat':
            # classification layer
            self.z_dropout = nn.Dropout(0.35)
            self.fc = nn.Linear(2*d, 1)

        # Apply weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        """
        Initializes the weights of the model layers.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def feature_extractor(self, x):
        """
        Extracts features from the input tensor, combining audio and visual features.

        Args:
            x (Tensor)      :   Input tensor of shape (batch_size, sequence_length, feature_dim).

        Returns:
            Tensor          :   Extracted features after fusion and encoding.
        """
        xa = x[:, :, 136:] # Audio # torch.Size([32, ~2259, 25]) 
        xv = x[:, :, :136] # Video # torch.Size([32, ~2259, 136])

        # GSABottleneckNet: Audio
        xa, gsa_cls = self.gsa_bottleneck(xa) # xa = torch.Size([32, 256, 71])

        # HATNET: Video
        xv = xv.permute(0, 2, 1)
        xv, hatnet_cls = self.hatnet(xv) # xv = torch.Size([32, 384, 71])

        # Downsample features to the target size
        xa = self.audio_downsample(xa).transpose(1, 2) # torch.Size([32, 139, 256])
        xv = self.video_downsample(xv).transpose(1, 2) # torch.Size([32, 139, 256])

        # # Cross the MT-1, MT-2 & T-1 transformer
        # xav_cross_feats = self.mutualtransformer(xv, xa) 

        if self.ablation =='add':
            ## Element wise adddistion
            x_ablation = xa + xv
            # print(f"x sum: {x_ablation.shape}") # x sum: torch.Size([8, 33, 256])

            # encoded the fused features
            xav_fused = self.av_encoder(x_ablation)
            # print(f"xav_fused_{self.ablation}: {xav_fused.shape}") # xav_fused: torch.Size([8, 33, 256])

        elif self.ablation == 'multi':
            ## Element wise multiplication
            x_ablation = xa * xv
            # print(f"x mul: {x_ablation.shape}") # x mul: torch.Size([8, 33, 256])

            # encoded the fused features
            xav_fused = self.av_encoder(x_ablation)
            # print(f"xav_fused_{self.ablation}: {xav_fused.shape}") # xav_fused: torch.Size([8, 33, 256])

        elif self.ablation == 'concat':
            ## Concatenation along the last dimension
            x_ablation = torch.cat((xa, xv), dim=-1)
            # print(f"x concat: {x_ablation.shape}") # x concat: torch.Size([8, 33, 512])

            # encoded the fused features
            xav_fused = self.av_encoder_concat(x_ablation)
            # print(f"xav_fused_{self.ablation}: {xav_fused.shape}") # xav_fused: torch.Size([8, 33, 512])
        
        z = torch.mean(xav_fused, dim=1)
        return self.z_dropout(z), gsa_cls, hatnet_cls

    def classifier(self, x):
        z, gsa_cls, hatnet_cls = x
        return self.fc(z), gsa_cls, hatnet_cls
        