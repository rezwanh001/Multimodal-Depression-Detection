# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
Adapted from: 
    (1) https://github.com/lucidrains/global-self-attention-network/blob/main/gsa_pytorch/gsa_pytorch.py
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from .base import BaseNet

def calc_reindexing_tensor(l, L, device):
    """
    Appendix B - (5)
    
    Calculates the reindexing tensor for relative positional encoding.
    
    Args:
        l (int)      : The sequence length.
        L (int)      : The relative positional length.
        device (str) : The device for tensor computation. 
    
    Returns:
        Tensor       : The reindexing tensor for relative positions.
    """
    x = torch.arange(l, device=device)[:, None, None]
    i = torch.arange(l, device=device)[None, :, None]
    r = torch.arange(-(L - 1), L, device=device)[None, None, :]
    mask = ((i - x) == r) & ((i - x).abs() <= L)
    return mask.float()

class GSA(nn.Module):
    def __init__(self, dim, *, rel_pos_length=None, dim_out=None, heads=8, dim_key=64, norm_queries=False, batch_norm=True):
        """
        Global Self-Attention (GSA) module with optional relative positional encoding.
        
        Args:
            dim (int)                  : The input feature dimension.
            rel_pos_length (int, opt.) : The length for relative positional encoding.
            dim_out (int, optional)    : The output feature dimension.
            heads (int)                : The number of attention heads.
            dim_key (int)              : The dimension of each attention head.
            norm_queries (bool)        : Whether to apply softmax normalization to queries.
            batch_norm (bool)          : Whether to apply batch normalization to relative positions.
        """
        super().__init__()
        
        # Set dim_out to dim if it's None
        self.dim_out = dim_out if dim_out is not None else dim
        
        dim_hidden = dim_key * heads

        self.heads = heads
        self.rel_pos_length = rel_pos_length
        self.norm_queries = norm_queries

        self.to_qkv = nn.Conv1d(dim, dim_hidden * 3, 1, bias=False)  # Conv1d for 1D data
        self.to_out = nn.Conv1d(dim_hidden, self.dim_out, 1)  # Conv1d for 1D data

        # Initialize relative positions if rel_pos_length is provided
        if rel_pos_length is not None:
            num_rel_shifts = 2 * rel_pos_length - 1
            self.norm = nn.BatchNorm1d(dim_key) if batch_norm else None  # BatchNorm1d for 1D data
            self.rel_positions = nn.Parameter(torch.randn(num_rel_shifts, dim_key))
        else:
            self.norm = None  # Set to None explicitly if rel_pos_length is None

    def forward(self, x):
        """
        Forward pass of the GSA module.
        
        Args:
            x (Tensor)  :   Input tensor of shape (batch_size, seq_len, dim).
        
        Returns:
            Tensor      :   Output tensor after applying global self-attention.
        """
        b, t, f, h, L, device = *x.shape, self.heads, self.rel_pos_length, x.device

        # Transpose to [batch, features, time] for Conv1d
        x = x.transpose(1, 2)

        # Apply convolution
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) t -> (b h) c t', h=h), qkv)

        # Softmax along the sequence length
        k = k.softmax(dim=-1)
        context = einsum('ndt,net->nde', k, v)

        content_q = q if not self.norm_queries else q.softmax(dim=-2)

        content_out = einsum('nde,ndt->net', context, content_q)
        content_out = rearrange(content_out, 'n d t -> n d t')

        # Relative positional encoding
        if self.rel_pos_length is not None:
            q, v = map(lambda t: rearrange(t, 'n c t -> n c t'), (q, v))

            It = calc_reindexing_tensor(t, L, device)
            Pt = einsum('tir,rd->tid', It, self.rel_positions)
            St = einsum('ndt,tid->nit', q, Pt)
            rel_pos_out = einsum('nit,net->net', St, v)

            # Apply normalization if norm is not None
            if self.norm is not None:
                rel_pos_out = self.norm(rel_pos_out)

            content_out = content_out + rel_pos_out.contiguous()

        content_out = rearrange(content_out, '(b h) c t -> b (h c) t', h=h)
        return self.to_out(content_out)
 
class DynamicBottleneck(nn.Module):
    def __init__(self, in_channels=25, out_channels=256, bottleneck_ratio=0.25):
        """
        Initializes the DynamicBottleneck module.
        
        Args:
            in_channels (int)       : Number of input channels.
            out_channels (int)      : Number of output channels.
            bottleneck_ratio (float): Ratio to reduce the number of channels in the bottleneck layer.
        """
        super(DynamicBottleneck, self).__init__()
        self.bottleneck_channels = int(in_channels * bottleneck_ratio)  # typically a smaller number

        # First conv layer: Reduce channels, keep sequence length the same
        self.conv1 = nn.Conv1d(in_channels, self.bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.bottleneck_channels)
        
        # Second conv layer: Change the sequence length (1398 -> 44)
        self.conv2 = nn.Conv1d(self.bottleneck_channels, self.bottleneck_channels, 
                               kernel_size=3, stride=32, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.bottleneck_channels)
        
        # Third conv layer: Expand channels to 256, sequence length should be 44 now
        self.conv3 = nn.Conv1d(self.bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to match the output shape
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=32, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels or 1398 != 44 else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the DynamicBottleneck.
        
        Args:
            x (Tensor)  : Input tensor of shape (batch_size, in_channels, seq_len).
        
        Returns:
            Tensor      : Output tensor of shape (batch_size, out_channels, new_seq_len).
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        # print(f"conv2 shape: {out.shape}")
        if out.size(2) > 1:  # Check if sequence length is greater than 1
            out = self.bn2(out)
            # print(f"bn2 shape: {out.shape}")
        else:
            print(f"Skipping BatchNorm1d (bn2) for input size {out.size()}") 

        # out = self.bn2(out)
        out = F.relu(out)
        # print(f"relu shape: {out.shape}")

        out = self.conv3(out)
        # print(f"conv3 shape: {out.shape}")
        if out.size(2) > 1:  # Check if the sequence length is greater than 1
            out = self.bn3(out)
            # print(f"bn3 shape: {out.shape}")
        else:
            print(f"Skipping BatchNorm1d (bn3) for input size {out.size()}")

        # out = self.bn3(out)

        out += self.shortcut(identity)
        # print(f"shortcut shape: {out.shape}")

        out = F.relu(out)
        # print(f"relu shape: {out.shape}")

        return out


class GSABottleneckNet(BaseNet):
    def __init__(self, hidden_sizes=[256, 256], dropout=0.5, gsa_input=25, gsa_rel_pos_length=10,):
        """
        Initializes the TMeanNetGSA model with the GSA module and fully connected layers.
        
        Args:
            hidden_sizes (list of int): Hidden layer sizes for the fully connected layers.
            dropout (float)           : Dropout rate for the fully connected layers.
            gsa_dim (int)             : The input dimension for the GSA module.
            gsa_heads (int)           : The number of attention heads in the GSA module.
            rel_pos_length (int)      : The relative positional length for the GSA module.
        """
        super().__init__()
        
        # Initialize GSA module
        self.gsa = GSA(dim=gsa_input, rel_pos_length=gsa_rel_pos_length)

        # apply bottleneck layer 
        self.bottleneck_layer = DynamicBottleneck(in_channels=25, out_channels=256, bottleneck_ratio=0.35)
        
        ## -------------------------
        # Add a small CNN or 1D convolution to process sequences
        self.conv1 = nn.Conv1d(25, 128, kernel_size=3, padding=1)  # 1D conv on variable length
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)

        # Adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool1d(96)  # Output fixed size [batch, 256, 96]

        # Fully connected layers
        self.classifier_gsa = nn.Sequential(
 
            # Fully connected layer after pooling
            nn.Linear(256 * 96, 128),  # Flattened input size after adaptive pooling
            nn.ReLU(),
            
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),  # Hidden layer
            nn.ReLU(),
            
            nn.Dropout(0.5),
            
            nn.Linear(64, 1)  # Output layer (e.g., for binary classification)
        )
        
        ##------------------------

        # self.bottleneck_layer = MultiHeadAttentionNet()
        
        # # Initialize fully connected layers
        # self.fcs = nn.Sequential()
        # last_dim = gsa_input
        # for h in hidden_sizes:
        #     self.fcs.append(nn.Linear(last_dim, h))
        #     self.fcs.append(nn.ReLU())
        #     self.fcs.append(nn.Dropout(dropout))
        #     last_dim = h

        # # Output layer
        # self.output = nn.Linear(last_dim, 1)

        # # classification layer
        # self.classifier_gsa = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(gsa_input, 1),  # Adjust this to match final feature size
        # )

        # # Classification layer with additional layers and activations
        # self.classifier_gsa = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
            
        #     # First fully connected layer
        #     nn.Linear(gsa_input, 128),  # Adjust hidden units as needed
        #     nn.ReLU(),
            
        #     # Second fully connected layer
        #     nn.Linear(128, 64),  # Adjust hidden units as needed
        #     nn.ReLU(),
            
        #     nn.Dropout(0.5),  # Additional dropout for regularization
            
        #     # Final classification layer
        #     nn.Linear(64, 1),  # Output layer for binary classification
        # )


        # Apply custom weight initialization
        self.apply(self.custom_weights_init)

    def custom_weights_init(self, m):
        """
        Custom weight initialization method.
        
        Args:
            m (nn.Module): A layer in the model.
        """
        if isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            init.normal_(m.data, mean=0.0, std=0.02)  # For additional parameters like self.rel_positions

    def feature_extractor(self, x):
        """
        Extracts features from the input using the GSA module.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim).
        
        Returns:
            Tensor: Feature tensor of shape (batch_size, dim).
        """
        # Apply GSA module to input
        gsa_out = self.gsa(x)
        # print(f"gsa_out: {gsa_out.shape}") # torch.Size([8, 25, ~1169])

        # gsa_out = gsa_out.permute(0, 2, 1)

        # gsa_bottleneck
        gsa_bottleneck = self.bottleneck_layer(gsa_out)
        # print(f"gsa_bottleneck shape: {gsa_bottleneck.shape}") # torch.Size([8, 256, ~91])
        
        # gsa_out_avg = F.adaptive_avg_pool1d(gsa_out, 1).flatten(1)

        ###-----------------------------------------------------------
        # Input shape: [batch_size, 25, variable length (~1169)]
        # First, swap dimensions to [batch_size, channels, sequence_length]
        # gsa_out = gsa_out.permute(0, 2, 1)  # Now shape: [batch_size, ~1169, 25]
        # print(f"--gsa_out: {gsa_out.shape}") 

        # Pass through conv layers
        gsa_out = F.relu(self.bn1(self.conv1(gsa_out)))  # Output shape: [batch_size, 128, ~1169]
        # print(f"1--gsa_out: {gsa_out.shape}") 
        gsa_out = F.relu(self.bn2(self.conv2(gsa_out)))  # Output shape: [batch_size, 256, ~1169]
        # print(f"2--gsa_out: {gsa_out.shape}") 

        # Apply adaptive average pooling to get a fixed size [batch_size, 256, 96]
        gsa_out = self.adaptive_pool(gsa_out)
        # print(f"3--gsa_out: {gsa_out.shape}") 

        # Flatten the output for the fully connected layers
        gsa_out = gsa_out.view(gsa_out.size(0), -1)  # Flatten to [batch_size, 256*96]
        # print(f"4--gsa_out: {gsa_out.shape}") 
        ##-------------------------------------------------------------

        # Mean pooling along the sequence length dimension
        return gsa_bottleneck, gsa_out # gsa_out_avg #gsa_out.mean(dim=2)  # mean along the sequence length

    def classifier(self, x):
        """
        Applies the fully connected layers and the output layer to the features.
        
        Args:
            x (Tensor): Input tensor after feature extraction.
        
        Returns:
            Tensor: Output tensor of shape (batch_size, 1).
        """
        gsa_bottleneck, gsa_out_mean = x
        # print(f"gsa_out_mean shape: {gsa_out_mean.shape}") # torch.Size([8, 25])

        return gsa_bottleneck,  self.classifier_gsa(gsa_out_mean) # self.output(self.fcs(gsa_out_mean)) #
