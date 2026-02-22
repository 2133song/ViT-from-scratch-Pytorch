import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
      def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
            super(PatchEmbedding, self).__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.n_patches = (img_size // patch_size) ** 2

            # Conv2d(kernel_size=patch_size, stride=patch_size)等价于分块+线性投影
            # 输出维度 = (输入维度-kernel_size+2*padding)/stride+1 = 输入维度/stride
            self.projection = nn.Conv2d(
                  in_channels,
                  embed_dim,
                  kernel_size=patch_size,
                  stride=patch_size
            )
      
      def forward(self, x):
            '''
                  x: (B, C, H, W) - 输入图像
                  Return: (B, N, D) - N个patch的embeddings
            '''
            x = self.projection(x)  # (B, 768, 14, 14)
            x = x.flatten(2)  # (B, 768, 196)
            x = x.transpose(1, 2)  # (B, 196, 768)
            return x
      