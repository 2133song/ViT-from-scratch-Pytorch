import torch
from models.vision_transformer import *
from vit_config import *
from models.patch_embedding import *
from models.msa import *
from models.transforemr_encoder_block import *
from util.data_augmentation import *
import matplotlib.pyplot as plt

def test_model_architecture():
      print("="*60)
      print("测试 ViT-Base/16 架构")
      print("="*60)

      model = VisionTransformer(**VIT_BASE_CONFIG)

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f'\n总参数量: {n_parameters:,} ({n_parameters/1e6:.2f}M)')
      print("\n各层参数分布:")
      print(f"Patch Embedding: {sum(p.numel() for p in model.patch_embed.parameters()):,}")
      print(f"Class Token: {model.cls_token.numel():,}")
      print(f"Position Embedding: {model.pos_embed.numel():,}")
      print(f"Transformer Blocks: {sum(p.numel() for p in model.blocks.parameters()):,}")
      print(f"Classification Head: {sum(p.numel() for p in model.head.parameters()):,}")
      
      print("\n" + "-"*60)
      print("测试前向传播")
      print("-"*60)

      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      model = model.to(device)
      model.eval()

      batch_size = 4
      x = torch.randn(batch_size, 3, 224, 224).to(device)
      print("\n输入形状：", x.shape)
      with torch.no_grad():
            output = model(x)
      print('输出形状：', output.shape)
      print('预期形状：', [batch_size, VIT_BASE_CONFIG['num_classes']])

      assert output.shape == (batch_size, VIT_BASE_CONFIG['num_classes']), "输出形状不匹配"
      print("前向传播测试通过！")
      return model

def test_patch_embedding():
      print("\n" + "="*60)
      print("测试 Patch Embedding")
      print("="*60)

      patch_embed = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)

      x = torch.randn(2, 3, 224, 224)
      patches = patch_embed(x)
      print('\n输入图像形状：', x.shape)
      print('输出patches形状：', patches.shape)
      print('预期：[2, 196, 768]')

      n_patches = (224 // 16) ** 2
      assert patches.shape[1] == n_patches, "patch数量错误"
      print('Patch Embedding测试通过！（Patch数量：）', n_patches)

def test_attention_mechanism():
      print("\n" + "="*60)
      print("测试 Multi-Head Self-Attention")
      print("="*60)

      attn = MultiHeadSelfAttention(dim=768, n_heads=12, dropout=0.1)

      x = torch.randn(2, 197, 768)  # (batch_size, num_tokens, embed_dim)
      with torch.no_grad():
            output = attn(x)
      print("\n输入形状：", x.shape)
      print('输出形状：', output.shape)
      assert output.shape == x.shape, "注意力输出形状与输入形状不同！"
      print('注意力机制测试通过！')

def test_transformer_block():
      print("\n" + "="*60)
      print("测试 Transformer Encoder Block")
      print("="*60)

      block = TransformerEncoderBlock(dim=768, n_heads=12, mlp_ratio=4, dropout=0.1)

      x = torch.randn(2, 197, 768)  # (batch_size, num_tokens, embed_dim)
      with torch.no_grad():
            output = block(x)
      print("\n输入形状：", x.shape)
      print('输出形状：', output.shape)
      assert output.shape == x.shape, "Transformer Encoder Block输出形状与输入形状不同！"
      print('Transformer块测试通过！')

def inference_example():
      print("\n" + "="*60)
      print("推理示例")
      print("="*60)

      device = 'cuda' if torch.cuda.is_available() else 'cpu'

      model = VisionTransformer(**VIT_BASE_CONFIG)
      model = model.to(device)
      model.eval()

      transform = get_transforms(img_size=224, is_training=False)
      batch_size = 8
      images = torch.randn(batch_size, 3, 224, 224).to(device)
      print(f"\n处理 {batch_size} 张图像...")

      with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # (batch_size, num_classes)
            top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
      for i in range(min(3, batch_size)):
            print('\n图像', i+1, ' :')
            for j in range(5):
                  print(f"Top-{j+1}: 类别 {top5_indices[i, j].item()}, "
                  f"概率 {top5_probs[i, j].item():.4f}")
      print('\n 推理完成！')

def main():
    print("\n" + "="*60)
    print("Vision Transformer (ViT-Base/16) 测试套件")
    print("="*60)
    
    # 测试各个组件
    test_patch_embedding()
    test_attention_mechanism()
    test_transformer_block()
    
    # 测试完整模型
    model = test_model_architecture()
    
    # 推理示例
    inference_example()
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)

if __name__ == '__main__':
    main()