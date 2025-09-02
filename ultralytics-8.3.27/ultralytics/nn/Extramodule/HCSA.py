

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

def channel_shuffle(x, groups=2):
    """Channel shuffle operation for mixing information across groups."""
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out

class HCSA(nn.Module):
    """
    Hybrid Channel-Spatial Attention (HCSA)
    

    """
    def __init__(self, c1, c2, reduction=4, num_heads=4, spatial_kernel_sizes=[3, 7]):
        super().__init__()
        assert c1 == c2, 
        
       
        self.branch_channels = c1  
        self.reduced_channels = max(32, c1 // reduction) 
        
       
        self.num_heads = max(1, min(num_heads, self.reduced_channels // 32))
        
       
        self.split_conv = Conv(c1, c1 * 2, 1)
        
     
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc1 = nn.Linear(self.branch_channels, self.reduced_channels)
        self.channel_ln = nn.LayerNorm(self.reduced_channels)
        self.channel_act = nn.SiLU(inplace=True)
        self.channel_fc2 = nn.Linear(self.reduced_channels, self.branch_channels)
        self.channel_sigmoid = nn.Sigmoid()
        
        self.spatial_h = nn.Conv2d(self.branch_channels, self.branch_channels, 
                                  kernel_size=(1, spatial_kernel_sizes[1]), 
                                  padding=(0, spatial_kernel_sizes[1]//2), 
                                  bias=False)
        self.spatial_v = nn.Conv2d(self.branch_channels, self.branch_channels, 
                                  kernel_size=(spatial_kernel_sizes[1], 1), 
                                  padding=(spatial_kernel_sizes[1]//2, 0), 
                                  bias=False)
        self.spatial_conv = Conv(self.branch_channels * 2, self.branch_channels, spatial_kernel_sizes[0])
        self.spatial_final = nn.Conv2d(self.branch_channels, self.branch_channels, kernel_size=1)
        self.spatial_sigmoid = nn.Sigmoid()
        
     
        self.qkv = nn.Linear(self.branch_channels, self.branch_channels * 3, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(self.branch_channels, self.branch_channels)
        self.proj_drop = nn.Dropout(0.1)
        
        
        self.use_pos_embed = False  
        
        
        self.ffn1 = Conv(self.branch_channels, self.branch_channels * 2, 1)
        self.ffn2 = Conv(self.branch_channels * 2, self.branch_channels, 1, act=False)
        
       
        self.cross_conv = Conv(self.branch_channels * 2, 2, 1, act=False)
        self.cross_sigmoid = nn.Sigmoid()
        
       
        self.output_conv = Conv(c1 * 2, c2, 1)

    def forward(self, x):
       
        split_features = self.split_conv(x)
        a, b = split_features.chunk(2, dim=1) 
        
       
        b_pool = self.channel_pool(b).flatten(1)  # [B, C]
        b_fc1 = self.channel_fc1(b_pool)  # [B, C_reduced]
        b_ln = self.channel_ln(b_fc1)
        b_act = self.channel_act(b_ln)
        b_fc2 = self.channel_fc2(b_act)  # [B, C]
        channel_attn = self.channel_sigmoid(b_fc2).view(b.shape[0], b.shape[1], 1, 1)
        b_channel = b * channel_attn
        
      
        spatial_h = self.spatial_h(b_channel)
        spatial_v = self.spatial_v(b_channel)
        spatial_combined = torch.cat([spatial_h, spatial_v], dim=1)
        spatial_conv = self.spatial_conv(spatial_combined)
        spatial_final = self.spatial_final(spatial_conv)
        spatial_attn = self.spatial_sigmoid(spatial_final)
        b_spatial = b_channel * spatial_attn
        
       
        a_shape = a.shape
        a_flat = a.flatten(2).transpose(1, 2)  # [B, HW, C]
        
       
        qkv = self.qkv(a_flat)  # [B, HW, 3*C]
        qkv = qkv.reshape(a_shape[0], -1, 3, self.num_heads, self.branch_channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, HW, C//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        
        x_attn = (attn @ v).transpose(1, 2).reshape(a_shape)
        a_attn = x_attn + a 
        
       
        a_ffn = self.ffn2(self.ffn1(a_attn))
        a_out = a_attn + a_ffn  
        
       
        combined = torch.cat([a_out, b_spatial], dim=1)
        gates = self.cross_sigmoid(self.cross_conv(combined))
        gate_a, gate_b = gates.chunk(2, dim=1)
        
        
        a_cross = a_out * gate_a + b_spatial * (1 - gate_a)  
        b_cross = b_spatial * gate_b + a_out * (1 - gate_b)  
        
       
        out = torch.cat([a_cross, b_cross], dim=1)
        
        
        out = channel_shuffle(out, groups=4)
        
       
        return self.output_conv(out)
