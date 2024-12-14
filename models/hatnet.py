# -*- coding: utf-8 -*-
'''
@author: Md Rezwanul Haque
Adapted from: https://github.com/yun-liu/HAT-Net/blob/main/HAT_Net.py
'''
#---------------------------------------------------------------
# Imports
#---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
#---------------------------------------------------------------
class InvertedResidual(nn.Module):
    """
    Defines the Inverted Residual block as used in many modern neural networks.

    Args:
        in_dim (int)                :   Number of input channels.
        hidden_dim (int, optional)  :   Number of hidden channels, defaults to in_dim.
        out_dim (int, optional)     :   Number of output channels, defaults to in_dim.
        kernel_size (int, optional) :   Size of the convolutional kernel, default is 3.
        drop (float, optional)      :   Dropout probability, default is 0.
        act_layer (class, optional) :   Activation function class, default is nn.SiLU.
    """
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2

        # First convolution block: Group normalization, pointwise convolution, and activation
        self.conv1 = nn.Sequential(
            nn.GroupNorm(1, in_dim, eps=1e-6),
            nn.Conv1d(in_dim, hidden_dim, 1, bias=False),
            act_layer(inplace=True)
        )

        # Second convolution block: Depthwise convolution with padding and activation
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad, groups=hidden_dim, bias=False),
            act_layer(inplace=True)
        )

        # Third convolution block: Pointwise convolution and group normalization
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim, eps=1e-6)
        )

        # Dropout layer
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        """
        Defines the forward pass of the Inverted Residual block.

        Args:
            x (torch.Tensor)    :   Input tensor with shape (batch_size, in_dim, length).

        Returns:
            torch.Tensor        :   Output tensor after applying the Inverted Residual block.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    """
    Attention mechanism with multi-head self-attention.

    This class implements a multi-head self-attention mechanism, including normalization, 
    linear projections for query, key, and value, and dropout regularization.

    Args:
        dim (int)                   :   Dimension of the input and output features.
        head_dim (int)              :   Dimension of each attention head.
        grid_size (int, optional)   :   Grid size for positional encoding (if applicable). Defaults to 1.
        ds_ratio (int, optional)    :   Downsampling ratio (if applicable). Defaults to 1.
        drop (float, optional)      :   Dropout rate. Defaults to 0.
    """
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, drop=0.):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size

        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.qkv = nn.Conv1d(dim, dim * 3, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.proj_norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        """
        Forward pass through the Attention mechanism.

        Args:
            x (torch.Tensor)    :   Input tensor with shape (batch_size, dim, sequence_length).

        Returns:
            torch.Tensor        :   Output tensor after applying the attention mechanism.
        """
        B, C, L = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, L)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(-2, -1).reshape(B, C, L)
        x = self.drop(self.proj(x))
        return x
    
class Block(nn.Module):
    """
    A building block combining attention and inverted residual convolution.

    This block applies a sequence of an attention mechanism followed by an inverted residual
    convolutional layer, both with optional dropout and drop path regularization.

    Args:
        dim (int)                       :   Dimension of the input and output features.
        head_dim (int)                  :   Dimension of each attention head.
        grid_size (int, optional)       :   Grid size for positional encoding (if applicable). Defaults to 1.
        ds_ratio (int, optional)        :   Downsampling ratio (if applicable). Defaults to 1.
        expansion (int, optional)       :   Expansion ratio for the hidden dimension in the inverted residual block. Defaults to 4.
        drop (float, optional)          :   Dropout rate applied after attention and convolution. Defaults to 0.
        drop_path (float, optional)     :   Drop path rate for stochastic depth regularization. Defaults to 0.
        kernel_size (int, optional)     :   Size of the convolution kernel in the inverted residual block. Defaults to 3.
        act_layer (nn.Module, optional) :   Activation function to use in the inverted residual block. Defaults to nn.SiLU.
    """
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, expansion=4,
                 drop=0., drop_path=0., kernel_size=3, act_layer=nn.SiLU):
        super().__init__()

        # Drop path for stochastic depth regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Attention mechanism
        self.attn = Attention(dim, head_dim, grid_size=grid_size, ds_ratio=ds_ratio, drop=drop)

        # Inverted residual block with convolution
        self.conv = InvertedResidual(dim, hidden_dim=dim * expansion, out_dim=dim,
            kernel_size=kernel_size, drop=drop, act_layer=act_layer)

    def forward(self, x):
        """
        Forward pass through the Block.

        The input tensor is processed through the attention mechanism and inverted residual block,
        with skip connections and optional drop path regularization applied after each stage.

        Args:
            x (torch.Tensor)    :   Input tensor with shape (batch_size, dim, sequence_length).

        Returns:
            torch.Tensor        :   Output tensor after processing through the block.
        """
        # Apply attention with skip connection and drop path
        x = x + self.drop_path(self.attn(x))

        # Apply inverted residual convolution with skip connection and drop path
        x = x + self.drop_path(self.conv(x))
        return x

class Downsample(nn.Module):
    """
    Downsample block for reducing the temporal dimension of the input.

    This block applies a 1D convolution with stride 2 to reduce the length of the input sequence
    by half, followed by group normalization for stabilizing the training.

    Args:
        in_dim (int)                :   Number of input channels.
        out_dim (int)               :   Number of output channels after downsampling.
        kernel_size (int, optional) :   Size of the convolution kernel. Defaults to 3.
    """
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        
        # 1D convolution for downsampling the input
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=1, stride=2)
        
        # Group normalization after convolution
        self.norm = nn.GroupNorm(1, out_dim, eps=1e-6)

    def forward(self, x):
        """
        Forward pass through the Downsample block.

        The input tensor is downsampled using convolution and then normalized.

        Args:
            x (torch.Tensor)        : Input tensor with shape (batch_size, in_dim, sequence_length).

        Returns:
            torch.Tensor            : Downsampled and normalized output tensor with shape 
                                        (batch_size, out_dim, sequence_length // 2).
        """
        x = self.norm(self.conv(x))
        return x


class HATNet(nn.Module):
    """
    HATNet: A hierarchical attention network for sequence classification.

    This network combines patch embedding, multiple stages of blocks with attention 
    and inverted residual layers, downsampling, and a final classification layer. 
    The architecture is designed to handle 1D sequence data, typically for tasks 
    such as time-series or audio classification.

    Args:
        seq_len (int, optional)             :   Length of the input sequences. Defaults to 1356.
        in_chans (int, optional)            :   Number of input channels. Defaults to 136.
        num_classes (int, optional)         :   Number of output classes for classification. Defaults to 1.
        dims (list of int, optional)        :   Dimensions of the feature maps in each stage. Defaults to [64, 128, 256, 512].
        head_dim (int, optional)            :   Dimension of each attention head. Defaults to 64.
        expansions (list of int, optional)  :   Expansion ratios for the inverted residual blocks in each stage. Defaults to [4, 4, 6, 6].
        grid_sizes (list of int, optional)  :   Grid sizes for positional encoding in each stage (if applicable). Defaults to [1, 1, 1, 1].
        ds_ratios (list of int, optional)   :   Downsampling ratios for each stage. Defaults to [8, 4, 2, 1].
        depths (list of int, optional)      :   Number of blocks in each stage. Defaults to [3, 4, 8, 3].
        drop_rate (float, optional)         :   Dropout rate applied in blocks and classifier. Defaults to 0.
        drop_path_rate (float, optional)    :   Drop path rate for stochastic depth regularization. Defaults to 0.
        act_layer (nn.Module, optional)     :   Activation function used in the inverted residual blocks. Defaults to nn.SiLU.
        kernel_sizes (list of int, optional):   Convolution kernel sizes for the inverted residual blocks in each stage. Defaults to [3, 3, 3, 3].
    """


    """  
    flowchart LR
        subgraph Patch Embedding
            A(Input) --> B(Conv1d) --> C(GroupNorm) --> D(Activation) --> E(Conv1d) --> F(Output)
        end
        subgraph Transformer Blocks
            G(Block 1) --> H(Block 2) --> I(Block 3) --> ... --> J(Block N)
        end
        F --> G
        
    """


    def __init__(self, seq_len=1356, in_chans=136, num_classes=1, dims=[64, 128, 256, 512],
                 head_dim=64, expansions=[4, 4, 6, 6], grid_sizes=[1, 1, 1, 1],
                 ds_ratios=[8, 4, 2, 1], depths=[3, 4, 8, 3], drop_rate=0.,
                 drop_path_rate=0., act_layer=nn.SiLU, kernel_sizes=[3, 3, 3, 3], 
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.depths = depths
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_chans, 16, 3, padding=1, stride=2),
            nn.GroupNorm(1, 16, eps=1e-6),
            act_layer(inplace=True),
            nn.Conv1d(16, dims[0], 3, padding=1, stride=2),
        )

        self.blocks = [] #nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for stage in range(len(dims)):
            blocks_stage = nn.ModuleList([
                Block(
                    dims[stage], head_dim, grid_size=grid_sizes[stage], ds_ratio=ds_ratios[stage],
                    expansion=expansions[stage], drop=drop_rate, drop_path=dpr[cur + i],
                    kernel_size=kernel_sizes[stage], act_layer=act_layer
                ).to(device)
                for i in range(depths[stage])
            ])
            self.blocks.append(blocks_stage)
            cur += depths[stage]

        self.ds2 = Downsample(dims[0], dims[1])
        self.ds3 = Downsample(dims[1], dims[2])
        self.ds4 = Downsample(dims[2], dims[3])
        
        # classification layer
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dims[-1], num_classes),  # Adjust this to match final feature size
        )

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initializes the weights of the network.

        Args:
            m (nn.Module)  :   A module from the network.
        """
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        """
        Forward pass through the HATNet.

        The input tensor is processed through patch embedding, multiple stages of blocks,
        downsampling layers, and finally a classifier.

        Args:
            x (torch.Tensor)        :   Input tensor with shape (batch_size, in_chans, seq_len).

        Returns:
            torch.Tensor            :   Output tensor with shape (batch_size, num_classes).
        """
        x = self.patch_embed(x)
      
        for block in self.blocks[0]:
            x = block(x)
        x = self.ds2(x)

        for block in self.blocks[1]:
            x = block(x)
        x = self.ds3(x)

        for block in self.blocks[2]:
            x = block(x)
        x = self.ds4(x)
        # print(x.shape)
        
        for block in self.blocks[3]:
            x3_block = block(x)
           
        '''
            3 block: torch.Size([4, 384, 43]) this will go to the transformer
        '''

        x_adAvg = F.adaptive_avg_pool1d(x3_block, 1).flatten(1)  # Changed to adaptive_avg_pool1d for 1D data
        # print(f"adaptive_avg_pool1d x_adAvg: {x_adAvg.shape}")

        x_hatnet_cls = self.classifier(x_adAvg)
        # print(f"classifier x_hatnet_cls: {x_hatnet_cls.shape}")

        return x3_block, x_hatnet_cls
