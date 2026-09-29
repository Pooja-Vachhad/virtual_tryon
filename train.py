import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from .dataset import train_loader , valid_loader
from .model import MobileNet
from tqdm import tqdm 
from .loss import DiceBCELoss , calculate_dice_score


checkpoint_path = "checkpoint.pth"
model_path = "best.pth"


def save_checkpoint(model , optimizer , scheduler , epoch , best_loss , counter , history , path = checkpoint_path):
   torch.save({
       "epoch":epoch ,
       "model": model.state_dict() ,
       "optimizer":optimizer.state_dict(),
       "scheduler":scheduler.state_dict(),
       "best_loss":best_loss ,
       "counter":counter,
       "history":history
   } ,path)


def empty_history():
    return {"train_loss": [], "valid_loss": [], "train_dice": [], "valid_dice": []}


def load_checkpoint(model , optimizer , scheduler , device , path= checkpoint_path):
   if os.path.exists(path):
     checkpoint = torch.load(path , map_location= device)
     model.load_state_dict(checkpoint["model"])
     optimizer.load_state_dict(checkpoint["optimizer"])
     scheduler.load_state_dict(checkpoint["scheduler"])
     print(f"resuming from epoch {checkpoint["epoch"] + 1}")
     return checkpoint["epoch"]+1 , checkpoint["best_loss"] , checkpoint["counter"] , checkpoint["history"]
   print("no checkpoint found , starting fresh")
   return 0 , float("inf") , 0 ,  empty_history()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNet(num_classes=1 , pretrained = True).to(device)

#quick shape test
model.eval()
with torch.no_grad():
  x = torch.randn(2 , 3 , 256 , 256).to(device)
  print(model(x).shape) #expect (2, 1, 256, 256)



optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = DiceBCELoss()   # sigmoid is indise this loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

def run_epoch(loader, model, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.set_grad_enabled(training):
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)

            output = model(image)          # raw logits, NO sigmoid here
            loss = criterion(output, mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += calculate_dice_score(output, mask)

    return total_loss / len(loader), total_dice / len(loader)





def run_model(train_loader, valid_loader, epochs, device, criterion, optimizer, model, scheduler, output_path, patience):
    start_epoch, best_loss, counter, history = load_checkpoint(model, optimizer, scheduler, device)

    for epoch in tqdm(range(start_epoch, epochs)):
        train_loss, train_dice = run_epoch(train_loader, model, criterion, device, optimizer)
        valid_loss, valid_dice = run_epoch(valid_loader, model, criterion, device)

        scheduler.step(valid_loss)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_dice"].append(train_dice)
        history["valid_dice"].append(valid_dice)

        # early stopping counter
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            counter += 1
            print(f"model not improving ({counter}/{patience})")

        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, counter, history)

        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
              f"train_dice={train_dice:.4f}, valid_dice={valid_dice:.4f}")

        if counter >= patience:
            print("patience exceeded, stopping early")
            break

    # ── plot loss curve ──
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["valid_loss"], label="valid_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ── plot dice curve ──
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_dice"], label="train_dice")
    plt.plot(history["valid_dice"], label="valid_dice")
    plt.xlabel("epochs")
    plt.ylabel("dice score")
    plt.title("Dice score curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return history

if __name__ == "__main__":
    history = run_model(train_loader, valid_loader, 20, device, criterion, optimizer, model, scheduler, model_path, 6)