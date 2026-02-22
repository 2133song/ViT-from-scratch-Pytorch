import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=768, n_heads=12, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** (-0.5)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
            x: (B, N, D) - N个token的embeddings
            Return: (B, N, D) - 注意力输出
        """
        B, N, D = x.shape  # (B, 196, 768)
        qkv = self.qkv(x)  # (B, 196, 3*768)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)  # (B, 196, 3, 12, 64)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, 12, 196, 64)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, 12, 196, 64)

        attention_score = (q @ k.transpose(-2, -1)) * self.scale  # (B, 12, 196, 196)
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)
        out = attention_score @ v  # (B, 12, 196, 64)

        out = out.transpose(1, 2)  # (B, 196, 12, 64)
        out = out.reshape(B, N, D)  # (B, 196, 768)
        out = self.proj(out)
        out = self.dropout(out)
        return out, attention_score
    