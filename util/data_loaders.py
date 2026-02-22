from torchvision import datasets
from data_augmentation import get_transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size, num_workers, img_size):
      train_transform = get_transforms(img_size, is_training=True)
      val_transform = get_transforms(img_size, is_training=False)

      train_dataset = datasets.ImageFolder(
            root = f"{data_dir}/train",
            transform = train_transform
      )
      val_dataset = datasets.ImageFolder(
            root = f"{data_dir}/val",
            transform = val_transform
      )

      train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
      )
      val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
      )
      return train_loader, val_loader