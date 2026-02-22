import torch

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
      model.train()
      total_loss = 0
      correct = 0
      total = 0

      for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if batch_idx % 100 == 0:
                  current_lr = optimizer.param_groups[0]['lr']
                  print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                        f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | '
                        f'LR: {current_lr:.6f}')
                  
      avg_loss = total_loss / len(train_loader)
      accuracy = 100. * correct / total

      return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
      model.eval()
      total_loss = 0
      correct = 0
      total = 0

      for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

      avg_loss = total_loss / len(val_loader)
      accuracy = 100. * correct / total

      return avg_loss, accuracy