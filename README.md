# Vision Transformer (ViT-Base/16) - 从零实现

基于论文 **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** 的完整PyTorch实现。

---

## 📋 目录

- [论文概述](#论文概述)
- [模型架构](#模型架构)
- [实现细节](#实现细节)
- [使用方法](#使用方法)
- [训练配置](#训练配置)
- [文件说明](#文件说明)
- [性能基准](#性能基准)

---

## 📚 论文概述

**论文标题**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
**作者**: Alexey Dosovitskiy et al. (Google Research)  
**发表**: ICLR 2021  
**链接**: https://arxiv.org/abs/2010.11929

### 核心思想

将图像视为一系列patches（类似于NLP中的tokens），直接应用标准Transformer架构进行图像分类。

### 关键创新

1. **图像分块**: 将224×224图像分割为196个16×16的patches
2. **线性投影**: 将每个patch展平并投影到D维空间
3. **位置编码**: 使用可学习的位置嵌入
4. **CLS Token**: 添加可学习的分类token用于分类任务
5. **纯Transformer**: 不使用卷积层，完全依赖自注意力机制

---

## 🏗️ 模型架构

### ViT-Base/16 配置

```
输入: 224×224×3 图像
├── Patch Embedding
│   ├── 16×16 patches → 196 patches
│   └── 线性投影: R^(16²×3) → R^768
├── Class Token: 1×768 (可学习)
├── Position Embedding: 197×768 (可学习)
├── Transformer Encoder ×12
│   ├── Multi-Head Self-Attention (12 heads)
│   │   ├── head_dim = 768/12 = 64
│   │   └── Scaled Dot-Product Attention
│   ├── Layer Norm (Pre-LN)
│   ├── MLP (768 → 3072 → 768)
│   └── Residual Connection
├── Layer Norm
└── Classification Head: 768 → num_classes
```

### 参数量

- **ViT-Base/16**: ~86M 参数
- **ViT-Large/16**: ~307M 参数
- **ViT-Huge/16**: ~632M 参数

---

## 🔧 实现细节

### 1. Patch Embedding

```python
# 论文公式: z_0 = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
```

**说明**:
- 使用卷积层实现patch分割+线性投影
- `kernel_size=stride=16` 确保无重叠的patches
- 输出: (B, 196, 768)

### 2. Multi-Head Self-Attention

```python
# 论文公式: Attention(Q,K,V) = softmax(QK^T / √d_k)V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=768, n_heads=12):
        self.qkv = nn.Linear(dim, dim * 3)  # Q, K, V投影
        self.scale = (dim // n_heads) ** -0.5  # 1/√d_k
```

**说明**:
- 12个注意力头，每个头维度64
- Scaled attention防止softmax饱和
- 多头并行计算后concat

### 3. Transformer Block

```python
# 论文公式:
# z'_l = MSA(LN(z_{l-1})) + z_{l-1}  (Pre-LN)
# z_l = MLP(LN(z'_l)) + z'_l

class TransformerEncoderBlock(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # MSA + residual
        x = x + self.mlp(self.norm2(x))   # MLP + residual
        return x
```

**说明**:
- Pre-LN架构（Layer Norm在注意力之前）
- 残差连接稳定训练
- MLP扩展比为4:1 (768→3072→768)

### 4. Position Embedding

```python
# 可学习的位置嵌入
self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
```

**说明**:
- 197个位置嵌入（196 patches + 1 class token）
- 论文实验表明可学习的效果优于固定的sin/cos
- 支持插值以适应不同分辨率

---

## 🚀 使用方法

### 安装依赖

```bash
pip install torch torchvision numpy
```

### 运行测试

```bash
# 测试模型架构和各个组件
python test_vit.py
```

---

## ⚙️ 训练配置

### 论文中的训练策略

#### 预训练（ImageNet-21k）

```python
{
    'dataset': 'ImageNet-21k',
    'images': 14_000_000,
    'classes': 21_000,
    'resolution': 224,
    'epochs': ~300,
    'batch_size': 4096,
    'optimizer': 'Adam',
    'base_lr': 3e-3,
    'weight_decay': 0.3,
    'warmup': '10k steps',
    'scheduler': 'cosine decay'
}
```

#### 微调（ImageNet-1k）

```python
{
    'dataset': 'ImageNet-1k',
    'images': 1_281_167,
    'classes': 1_000,
    'resolution': 224 or 384,
    'epochs': 20-50,
    'batch_size': 512,
    'base_lr': 3e-4,  # 降低10倍
    'weight_decay': 0.3
}
```

### 数据增强

```python
# 训练时
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 推理时
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### 学习率调度

```
Warmup阶段 (0-10k steps):
    lr = base_lr * (current_step / warmup_steps)

Cosine Decay阶段 (10k-total steps):
    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

---

## 📁 文件说明

```
.
├── models
│   ├── patch_embedding.py           # Patch嵌入层
│   ├── msa.py                       # 多头自注意力
│   ├── mlp.py                       # 前馈网络
│   ├── transformer_encoder_block.py # Transformer块
│   ├── vision_transformer.py        # 完整模型
│
├── util             
│   ├── data_augmentation.py         # 数据增强
│   ├── data_loaders.py              # 数据加载
│   └── train_scheduler.py           # 训练策略
│
├── engine.py            
│   ├── train_one_epoch
|   ├── evaluate
|
├── train.py            
│   ├── train_vit
│
├── vit_config.py             
│   ├── VIT_BASE_CONFIG              # Base变体配置
│   ├── VIT_LARGE_CONFIG             # Large变体配置
│   ├── VIT_HUGE_CONFIG              # Huge变体配置
│   └── TRAINING_CONFIG              # 训练超参数
│
├── test_vit.py   
│   ├── test_model_architecture
│   ├── test_patch_embedding
│   ├── test_attention_mechanism
│   └── inference_example
│
└── README.md                        # 本文档
```

## 📊 性能基准

### ImageNet-1k结果（论文报告）

| Model | Params | Pre-train | Resolution | Top-1 Acc |
|-------|--------|-----------|------------|-----------|
| ViT-Base/16 | 86M | ImageNet-21k | 224 | 77.9% |
| ViT-Base/16 | 86M | ImageNet-21k | 384 | 79.7% |
| ViT-Large/16 | 307M | ImageNet-21k | 224 | 82.6% |
| ViT-Large/16 | 307M | ImageNet-21k | 384 | 84.2% |
| ViT-Huge/14 | 632M | JFT-300M | 224 | 85.1% |

### 训练成本

- **ViT-Base**: 8×V100 GPUs, ~7天
- **ViT-Large**: 8×V100 GPUs, ~14天
- **ViT-Huge**: 大规模TPU集群

---

## 🎯 后续改进方向

1. **DeiT**: 数据高效的训练策略
2. **Swin Transformer**: 层次化架构
3. **BEiT**: 自监督预训练
4. **MAE**: Masked Autoencoder
5. **ViT-G**: 更大的模型规模

---

## 📖 参考资料

- **原始论文**: https://arxiv.org/abs/2010.11929
- **官方代码**: https://github.com/google-research/vision_transformer
- **Attention机制**: https://arxiv.org/abs/1706.03762
- **ImageNet数据集**: https://www.image-net.org/

---

## 🤝 贡献

欢迎提出问题和改进建议！

---

## 📄 许可

本实现仅用于学习和研究目的。

---

## ✨ 致谢

感谢Google Research团队的开创性工作，以及PyTorch社区的支持。
