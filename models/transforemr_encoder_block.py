import torch
import torch.nn as nn
from models.msa import *
from models.mlp import *

class TransformerEncoderBlock(nn.Module):
      def __init__(self, dim=768, n_heads=12, mlp_ratio=4, dropout=0.1):
            super(TransformerEncoderBlock, self).__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attention = MultiHeadSelfAttention(dim, n_heads, dropout)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = MLP(dim, dim*mlp_ratio, dropout)

      def forward(self, x):
            _x = x
            x = self.norm1(x)  # 这里采用pre-norm，Transformer原论文采用post-norm
            x, _  = self.attention(x)
            x = x + _x
            x = x + self.mlp(self.norm2(x))
            return x
      