import torch
import torch.nn as nn

class MLP(nn.Module):
      def __init__(self, dim=768, hidden_dim=768*4, dropout=0.1):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.GELU()  # 原论文中使用GLUE激活函数

      def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x