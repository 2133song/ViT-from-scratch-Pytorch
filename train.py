import torch
import torch.nn as nn
import numpy as np
from engine import *
from util.train_scheduler import *
from models.vision_transformer import *
from util.data_loaders import *

def train_vit(
      model, 
      train_loader,
      val_loader,
      epochs=300,
      base_lr=3e-3,
      weight_decay=0.3,
      warmup_epochs=10,
      device='cuda'
):
      model = model.to(device)
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
      )
      steps_per_epoch = len(train_loader)
      total_steps = epochs * steps_per_epoch
      warmup_steps = warmup_epochs * steps_per_epoch
      scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=1e-5
      )

      best_acc = 0.0
      for epoch in range(epochs):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'{"="*60}')
            train_loss, train_acc = train_one_epoch(
                  model, train_loader, criterion, optimizer, scheduler, device, epoch
            )
            val_loss, val_acc = evaluate(
                  model, val_loader, criterion, device
            )
            print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            if val_acc > best_acc:
                  best_acc = val_acc
                  torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc
                  }, 'vit_base_best.pth')
                  print(f'Best model saved! (Acc: {best_acc:.2f}%)')
      return model

def main():
      torch.manual_seed(42)
      np.random.seed(42)

      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      print(f'Using device: {device}')

      # ViT-Base/16 模型配置
      model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=1000,  # ImageNet-1k
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            emb_dropout=0.1
      )

      n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
      print(f'\nModel: ViT-Base/16')
      print(f'Number of parameters: {n_parameters/1e6:.2f}M')

      data_dir = './'
      train_loader, val_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=128,
            num_workers=4,
            img_size=224
      )
      model = train_vit(
      model=model,
      train_loader=train_loader,
      val_loader=val_loader,
      epochs=300,
      base_lr=3e-3,
      weight_decay=0.3,
      warmup_epochs=10,
      device=device
      )

if __name__ == '__main__':
      main()