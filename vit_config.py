# ViT-Base/16 配置
VIT_BASE_CONFIG = {
      'img_size': 224,
      'patch_size': 16,
      'in_channels': 3,
      'num_classes': 1000,
      'embed_dim': 768,
      'depth': 12,
      'n_heads': 12,
      'mlp_ratio': 4,
      'dropout': 0.1,
      'emb_dropout': 0.1
}

# ViT-Large/16 配置
VIT_LARGE_CONFIG = {
      'img_size': 224,
      'patch_size': 16,
      'in_channels': 3,
      'num_classes': 1000,
      'embed_dim': 1024,
      'depth': 24,
      'n_heads': 16,
      'mlp_ratio': 4,
      'dropout': 0.1,
      'emb_dropout': 0.1
}

# ViT-Huge/16 配置
VIT_HUGE_CONFIG = {
      'img_size': 224,
      'patch_size': 16,
      'in_channels': 3,
      'num_classes': 1000,
      'embed_dim': 1200,
      'depth': 32,
      'n_heads': 16,
      'mlp_ratio': 4,
      'dropout': 0.1,
      'emb_dropout': 0.1
}

TRAINING_CONFIG = {
      'optimizer': 'adam',
      'base_lr': 3e-3,
      'betas': (0.9, 0.999),
      'weight_decay': 0.3,

      'warmup_epochs': 10,
      'scheduler': 'cosine',
      'min_lr': 1e-5,

      'epochs': 300,
      'batch_size': 128,
      'gradient_clip': 1.0,

      'random_crop_scale': (0.08, 1.0),
      'horizontal_flip': True,

      'num_workers': 4,
      'pin_memory': True,
      'seed': 42
}
