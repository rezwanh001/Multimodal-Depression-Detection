# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
Source Paper: https://arxiv.org/abs/2401.14185
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------

import torch
import torch.nn as nn

class MutualTransformer(nn.Module):
    def __init__(self, d=256, f_d=512, l=6):
        """
        Initializes the MutualTransformer class.

        Args:
            d (int)                 :   Dimensionality of the model, used as the `d_model` parameter in the Transformer layers.
            f_d (int)               :   Fused Dimension from the model
            l (int)                 :   Number of layers in the Transformer encoder.
            num_iterations (int)    :   Number of cross-attention iterations between audio and visual features.
        """
        super().__init__()

        # Encoders for visual features
        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l,
        )

        # Encoders for audio features
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d, batch_first=True
            ),
            num_layers=l,
        )

        # Projection layers for cross-attention
        ## Audio
        self.qa_transform = nn.Linear(d, d)
        self.ka_transform = nn.Linear(d, d)
        self.va_transform = nn.Linear(d, d)
        ## Video
        self.qv_transform = nn.Linear(d, d)
        self.kv_transform = nn.Linear(d, d)
        self.vv_transform = nn.Linear(d, d)

        ## Fused: Audio + Video
        self.qf_transform = nn.Linear(f_d, d)
        self.kf_transform = nn.Linear(f_d, d)
        self.vf_transform = nn.Linear(f_d, d)

        # Cross-attention layers: audio x video
        self.cross_av = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )

        # Cross-attention layers: video x audio
        self.cross_va = nn.MultiheadAttention(
            embed_dim=d, num_heads=4, batch_first=True
        )

    def forward(self, v, a):
        """
        Forward pass for the MutualTransformer model.

        Args:
            v (torch.Tensor)        :   Input tensor representing visual features, shape [batch_size, seq_length, d].
            a (torch.Tensor)        :   Input tensor representing audio features, shape [batch_size, seq_length, d].

        Returns:
            torch.Tensor            :   Output tensor after mutual cross-attention and fusion, shape [batch_size, seq_length, 2*d].
        """
        # Encode visual and audio features
        v_encoded = self.v_encoder(v)
        a_encoded = self.a_encoder(a)

        # MT-1: Audio (q), Video (v, k)
        q_a = self.qa_transform(a_encoded)
        k_a = self.ka_transform(v_encoded)
        v_a = self.va_transform(v_encoded)
        # Cross Attention    
        fav = self.cross_av(q_a, k_a, v_a)[0]

        # MT-2: Video (q), Audio (v, k)
        q_v = self.qv_transform(v_encoded)
        k_v = self.kv_transform(a_encoded)
        v_v = self.vv_transform(a_encoded)
        # Cross Attention
        fva = self.cross_va(q_v, k_v, v_v)[0]

        # T-3: Audio + Video Features
        fused_a_v = torch.cat([v_encoded, a_encoded], dim=2)
        # print(f"fused_a_v: {fused_a_v.shape}")
        q_f = self.qf_transform(fused_a_v)
        k_f = self.kf_transform(fused_a_v)
        v_f = self.vf_transform(fused_a_v)
        # Cross Attention
        f_a_v = self.cross_va(q_f, k_f, v_f)[0]

        # Concatenate and encode fused features: M-1, M-2 & T-1
        fused_features = torch.cat([fav, fva, f_a_v], dim=2)
        
        return fused_features