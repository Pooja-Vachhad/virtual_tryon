"""
test.py

Load the best saved model and run it on the test set.
1. Prints the Dice+BCE loss AND the dice score over the WHOLE test set.
2. Shows n_size samples: original image, ground truth mask, predicted mask.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .model import MobileNet
from tqdm import tqdm 
from .loss import DiceBCELoss , calculate_dice_score
from .dataset import test_loader

# same values used in Normalize() in the transforms
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(img):
    img = img.cpu() * STD + MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def evaluate_test_set(loader, model, device):
    criterion = DiceBCELoss()
    model.eval()
    total_loss, total_dice, total_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)                      # logits
            loss = criterion(output, masks)
            dice = calculate_dice_score(output, masks)

            total_loss += loss.item() * images.size(0)   # weight by batch size
            total_dice += dice * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples, total_dice / total_samples



def show_predictions(loader, model, n_size, device):
    model.eval()
    fig, axes = plt.subplots(n_size, 3, figsize=(10, 4 * n_size), squeeze=False)
    shown = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            output = model(images)
            pred_mask = (torch.sigmoid(output) > 0.5).float()  # sigmoid + threshold -> 0/1 mask

            for i in range(images.size(0)):
                if shown == n_size:
                    break

                axes[shown, 0].imshow(denormalize(images[i]))
                axes[shown, 0].set_title("original image")

                axes[shown, 1].imshow(masks[i, 0].cpu().numpy(), cmap="gray")
                axes[shown, 1].set_title("ground truth mask")

                axes[shown, 2].imshow(pred_mask[i, 0].cpu().numpy(), cmap="gray")
                axes[shown, 2].set_title("predicted mask")

                for ax in axes[shown]:
                    ax.axis("off")
                shown += 1

            if shown == n_size:
                break

    plt.tight_layout()
    plt.show()



model = MobileNet(num_classes=1 , pretrained = False).to(device)
model.load_state_dict(torch.load("best.pth", map_location=device))
model.eval()

test_loss, test_dice = evaluate_test_set(test_loader, model, device)
print(f"Test loss: {test_loss:.4f} | Test dice: {test_dice:.4f}")

show_predictions(test_loader, model, 5, device)