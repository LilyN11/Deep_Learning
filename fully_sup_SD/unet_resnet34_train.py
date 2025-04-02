import time
import performance_metrics as pm
import torch

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    import time
    model.train()
    
    total_loss = 0
    previous_loss = None
    num_batches = len(dataloader)
    timer = time.time()

    for i_batch, (images, masks) in enumerate(dataloader):
        batch_start = time.time()
        if previous_loss is None:
            print(f"Running batch {i_batch+1}/{num_batches}, batch time {batch_start - timer:.2f}s", end="\r")
        else:
            print(f"Running batch {i_batch+1}/{num_batches}, batch time {batch_start - timer:.2f}, batch loss {previous_loss:.4f}", end="\r")

        timer = batch_start

        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        previous_loss = loss_value

        # if i_batch == 2: break

    avg_loss = total_loss / (i_batch + 1)  # Only count batches actually run
    print()
    return avg_loss

def train_epochs(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, save_checkpoints=True, checkpoint_name=None):
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{10}")
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print("Evaluating...")
        val_iou, val_dice = pm.evaluate(model, val_loader, device)
        print("Evaluation complete.")

        print(f"Epoch time: {time.time() - start_time:.2f}s")
        start_time = time.time()

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        if save_checkpoints: 
            file_name = f"checkpoints/unet_epoch{epoch+1}_"+checkpoint_name+".pth"
            torch.save(model.state_dict(), file_name)

    return val_iou, val_dice
