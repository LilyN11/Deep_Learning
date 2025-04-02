import torch
import matplotlib.pyplot as plt
import numpy as np


def denormalize(tensor, mean, std):
    """
    Reverses normalization so the image can be displayed properly.
    Args:
        tensor: [3, H, W] or [H, W, 3]
        mean, std: same as used in Normalize
    Returns:
        numpy image in range [0, 1]
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.clone().detach().cpu()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        tensor = tensor.permute(1, 2, 0).numpy()
    elif isinstance(tensor, np.ndarray):
        for c in range(3):
            tensor[..., c] = tensor[..., c] * std[c] + mean[c]
    return np.clip(tensor, 0, 1)


def visualise_predictions(model, dataloader, device, num_images=5, threshold=0.5, binarize_masks=True, use_color_for_truth=False):
    """
    Visualizes model predictions and ground truth masks.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader (e.g., val_loader)
        device: torch.device
        num_images: Number of examples to show
        threshold: Threshold on sigmoid for binary masks
        binarize_masks: If True, treat ground truth as binary (mask == 1)
        use_color_for_truth: If True, color-code ground truth with 'tab10'
    """
    model.eval()
    shown = 0
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            preds = model(images)
            preds = torch.sigmoid(preds) > threshold  # binary prediction

            for i in range(images.size(0)):
                if shown >= num_images:
                    return

                # De-normalize image for display
                img = denormalize(images[i], mean=imagenet_mean, std=imagenet_std)

                # Predicted and true masks
                pred = preds[i][0].cpu().numpy()
                gt = masks[i][0].cpu().numpy()

                # Prepare ground truth for display
                if binarize_masks:
                    gt_display = (gt == 1).astype(np.float32)
                elif use_color_for_truth:
                    gt_display = gt
                else:
                    gt_display = gt * 85  # scale 1→85, 2→170, 3→255

                # --- Plotting ---
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title("Image")
                axes[0].axis("off")

                axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Predicted Mask")
                axes[1].axis("off")

                if binarize_masks:
                    axes[2].imshow(gt_display, cmap="gray", vmin=0, vmax=1)
                elif use_color_for_truth:
                    axes[2].imshow(gt_display, cmap="tab10", vmin=1, vmax=3)
                else:
                    axes[2].imshow(gt_display, cmap="gray", vmin=0, vmax=255)
                axes[2].set_title("Ground Truth Mask")
                axes[2].axis("off")

                plt.tight_layout()
                plt.show()
                plt.close()

                shown += 1