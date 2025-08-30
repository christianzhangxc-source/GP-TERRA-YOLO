# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
MR_HCSA (递归岩石纹理增强版HCSA)
结合递归通道注意力和轻量级岩石纹理编码
"""

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

class LightweightTextureEnhancer(nn.Module):
    """超轻量级岩石纹理增强器"""
    def __init__(self, channels):
        super().__init__()
        # 使用深度可分离卷积极大减少计算量
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2, 
            groups=channels, bias=False  # 每个通道单独卷积
        )
        self.pointwise = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.activate = nn.SiLU(inplace=True)
        
        # 可学习缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # 深度可分离卷积
        texture = self.depthwise(x)
        texture = self.pointwise(texture)
        texture = self.activate(texture)
        
        # 残差连接 + 可学习缩放
        return x + self.gamma * texture

class RecursiveChannelAttention(nn.Module):
    """递归通道注意力 - 在单模块内模拟多层堆叠效果"""
    def __init__(self, channels, reduced_channels, recursion_depth=3):
        super().__init__()
        self.recursion_depth = recursion_depth
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 通道降维
        self.down = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.LayerNorm(reduced_channels),
            nn.SiLU(inplace=True)
        )
        
        # 递归处理单元
        self.recursive_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(reduced_channels, reduced_channels),
                nn.LayerNorm(reduced_channels),
                nn.SiLU(inplace=True)
            ) for _ in range(recursion_depth)
        ])
        
        # 通道升维
        self.up = nn.Linear(reduced_channels, channels)
        self.sigmoid = nn.Sigmoid()
        
        # 递归步数的权重
        self.recursive_weights = nn.Parameter(torch.ones(recursion_depth + 1))
        
    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        
        # 初始特征
        feat = self.down(self.pool(x).flatten(1))
        
        # 保存所有递归步骤的结果
        recursive_feats = [feat]
        
        # 递归细化
        for layer in self.recursive_layers:
            feat = layer(feat)
            recursive_feats.append(feat)
        
        # 自适应加权组合不同递归深度的特征
        weights = F.softmax(self.recursive_weights, dim=0)
        combined = torch.zeros_like(recursive_feats[0])
        for i, feat in enumerate(recursive_feats):
            combined += feat * weights[i]
        
        # 生成注意力权重
        channel_attn = self.sigmoid(self.up(combined)).view(b, c, 1, 1)
        
        return channel_attn


class MR_HCSA(nn.Module):
    """
    递归岩石纹理增强型 HCSA
    
    结合递归通道注意力和轻量级岩石纹理编码
    """
    def __init__(self, c1, c2, reduction=4, num_heads=4, spatial_kernel_sizes=[3, 7], recursion_depth=3):
        super().__init__()
        assert c1 == c2, "输入输出通道数必须相同以维持YOLOv8结构"
        
        # 轻量级岩石纹理增强器 - 用在早期特征增强
        self.texture_enhancer = LightweightTextureEnhancer(c1)
        
        # 参数定义
        self.branch_channels = c1  # 每个分支的通道数
        self.reduced_channels = max(32, c1 // reduction)  # 确保至少有32个通道
        self.num_heads = max(1, min(num_heads, self.reduced_channels // 32))
        
        # 输入分割
        self.split_conv = Conv(c1, c1 * 2, 1)
        
        #===== 第一分支: GAM路径 =====
        # 递归通道注意力
        self.channel_attention = RecursiveChannelAttention(
            self.branch_channels, self.reduced_channels, recursion_depth
        )
        
        # 方向感知空间注意力 - 保持原HCSA实现
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
        
        #===== 第二分支: PSA路径 - 保持原HCSA实现 =====
        self.qkv = nn.Linear(self.branch_channels, self.branch_channels * 3, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(self.branch_channels, self.branch_channels)
        self.proj_drop = nn.Dropout(0.1)
        
        # 位置编码设置 - 保持与原HCSA一致
        self.use_pos_embed = False
        
        # FFN - 保持原HCSA实现
        self.ffn1 = Conv(self.branch_channels, self.branch_channels * 2, 1)
        self.ffn2 = Conv(self.branch_channels * 2, self.branch_channels, 1, act=False)
        
        # 流交互注意力 - 保持原HCSA实现
        self.cross_conv = Conv(self.branch_channels * 2, 2, 1, act=False)
        self.cross_sigmoid = nn.Sigmoid()
        
        # 输出融合 - 保持原HCSA实现
        self.output_conv = Conv(c1 * 2, c2, 1)

    def forward(self, x):
        # 岩石纹理增强 - 在输入阶段应用
        x = self.texture_enhancer(x)
        
        # 特征分割
        split_features = self.split_conv(x)
        a, b = split_features.chunk(2, dim=1)  # 分成两个分支，每个分支的通道数为c1
        
        #===== 第一分支: GAM路径处理 =====
        # 递归通道注意力
        channel_attn = self.channel_attention(b)
        b_channel = b * channel_attn
        
        # 方向感知空间注意力 - 与原HCSA相同
        spatial_h = self.spatial_h(b_channel)
        spatial_v = self.spatial_v(b_channel)
        spatial_combined = torch.cat([spatial_h, spatial_v], dim=1)
        spatial_conv = self.spatial_conv(spatial_combined)
        spatial_final = self.spatial_final(spatial_conv)
        spatial_attn = self.spatial_sigmoid(spatial_final)
        b_spatial = b_channel * spatial_attn
        
        #===== 第二分支: PSA路径 - 完全按照原HCSA实现 =====
        a_shape = a.shape
        a_flat = a.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # 生成 q, k, v - 与原HCSA相同
        qkv = self.qkv(a_flat)  # [B, HW, 3*C]
        qkv = qkv.reshape(a_shape[0], -1, 3, self.num_heads, self.branch_channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, HW, C//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力 - 与原HCSA相同
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 不使用位置编码 - 与当前HCSA实现一致
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力 - 与原HCSA相同
        x_attn = (attn @ v).transpose(1, 2).reshape(a_shape)
        a_attn = x_attn + a  # 第一个残差连接
        
        # FFN处理 + 残差连接 - 与原HCSA相同
        a_ffn = self.ffn2(self.ffn1(a_attn))
        a_out = a_attn + a_ffn  # 第二个残差连接
        
        #===== 流交互和输出处理 - 与原HCSA相同 =====
        # 流交互注意力
        combined = torch.cat([a_out, b_spatial], dim=1)
        gates = self.cross_sigmoid(self.cross_conv(combined))
        gate_a, gate_b = gates.chunk(2, dim=1)
        
        # 交叉注意力加权
        a_cross = a_out * gate_a + b_spatial * (1 - gate_a)  # 将空间信息融入通道流
        b_cross = b_spatial * gate_b + a_out * (1 - gate_b)  # 将通道信息融入空间流
        
        # 特征融合
        out = torch.cat([a_cross, b_cross], dim=1)
        
        # 通道混洗增强特征交互
        out = channel_shuffle(out, groups=4)
        
        # 输出处理
        return self.output_conv(out)