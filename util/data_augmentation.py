from torchvision import transforms

def get_transforms(img_size, is_training=True):
      '''
            Training: Random crop(随机裁剪) + Horizontal flip(水平翻转)
            Inference: Resize to 256, center crop to 224
      '''
      if is_training:
            data_transforms = transforms.Compose([
                  transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
            ])
            return data_transforms
      else:
            data_transforms = ([
                  transforms.Resize(256),
                  transforms.CenterCrop(img_size),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
            ])
            return data_transforms