import torch
import torch.nn as nn
from models.patch_embedding import *
from models.transforemr_encoder_block import *

class VisionTransformer(nn.Module):
      def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, 
                   depth=12, n_heads=12, mlp_ratio=4, dropout=0.1, emb_dropout=0.1):
            super(VisionTransformer, self).__init__()
            self.num_classes = num_classes
            self.embed_dim = embed_dim

            self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
            n_patches = self.patch_embed.n_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, 768)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+1, embed_dim))  # (1, 197, 768)
            self.pos_drop = nn.Dropout(emb_dropout)

            self.blocks = nn.ModuleList([
                  TransformerEncoderBlock(embed_dim, n_heads, mlp_ratio, dropout)
                  for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
            self.head = nn.Linear(embed_dim, num_classes)

            self._init_weights()

      def _init_weights(self):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            for m in self.modules():
                  if isinstance(m, nn.Linear):
                        nn.init.trunc_normal_(m.weight, std=0.02)
                        if m.bias is not None:
                              nn.init.constant_(m.bias, 0)
                  elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.weight, 1.0)
      
      def forward(self, x):
            '''
                  x: (B, 3, 224, 224) - 输入图像
                  Return: (B, num_classes) - 分类logits
            '''
            B = x.shape[0]
            x = self.patch_embed(x)  # (B, 196, 768)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            for block in self.blocks:
                  x = block(x)
            x = self.norm(x)
            cls_output = x[:, 0]  # (B, 768)，第一个token
            logits = self.head(cls_output)
            return logits
