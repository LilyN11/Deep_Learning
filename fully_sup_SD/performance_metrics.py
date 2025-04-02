import torch

def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def compute_dice(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()

def evaluate(model, dataloader, device):
    with torch.no_grad():
        model.eval()
        
    iou_scores, dice_scores = [], []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images).squeeze(1)

            for p, t in zip(preds, masks):
                iou_scores.append(compute_iou(p, t))
                dice_scores.append(compute_dice(p, t))

    return sum(iou_scores) / len(iou_scores), sum(dice_scores) / len(dice_scores)

