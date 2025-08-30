# ultralytics/nn/modules/attention.py

import math
import torch
import torch.nn as nn
from .conv import Conv

class PSAttention(nn.Module):
    def __init__(self, c1, c2, e=0.5, num_heads=4):
        super().__init__()
        assert(c1 == c2)  # 保持YOLOv8的输入输出通道相同的要求
        
        # 使用通道缩放因子计算内部通道数
        self.c = int(c1 * e)
        
        # 使用Conv从YOLOv8而不是直接的nn.Conv2d
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 输入分割卷积
        self.cv2 = Conv(2 * self.c, c1, 1)     # 输出融合卷积
        
        # 确保头数与通道数兼容
        self.num_heads = min(num_heads, self.c // 32)  # 防止头数过多
        if self.num_heads == 0:
            self.num_heads = 1
        
        # 注意力机制相关层
        self.qkv = nn.Linear(self.c, self.c * 3, bias=False)
        self.proj = nn.Linear(self.c, self.c)
        
        # FFN
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        # 分割特征
        x_split = self.cv1(x)
        a, b = x_split.split((self.c, self.c), dim=1)
        
        # 多头自注意力处理
        b_shape = b.shape
        b_flat = b.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # 生成q, k, v
        qkv = self.qkv(b_flat)  # [B, HW, 3*C]
        qkv = qkv.reshape(b_shape[0], -1, 3, self.num_heads, self.c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, HW, C//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.c // self.num_heads)  # 缩放点积注意力
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(b_shape[0], -1, b_shape[2], b_shape[3])
        out = self.proj(out.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b_shape)
        
        # 残差连接
        b = b + out  # 第一个残差连接
        
        # FFN处理 + 残差连接
        b = b + self.ffn(b)  # 第二个残差连接
        
        # 融合特征
        return self.cv2(torch.cat((a, b), 1))