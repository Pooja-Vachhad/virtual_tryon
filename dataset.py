import os
import torch
import torch.nn as nn
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .model import MobileNet
from .loss import DiceBCELoss , calculate_dice_score


#set the folder paths
image_path = "path to image dir"
mask_path = "path to the mask dir"


#set the global seed so runs are reproducible
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


#match each image with its mask file
def folder_path(image_dir , mask_dir):
   images =[]
   masks = []
   skipped =[]

   for img in sorted(os.listdir(image_dir)):
        mask_name = img.replace("image" , "mask").replace(".jpg" , ".png")
        mask_file = os.path.join(mask_dir , mask_name)

        if os.path.exists(mask_file):
            images.append(os.path.join(image_dir , img))
            masks.append(mask_file)
        else:
            skipped.append(mask_file) #skip if no mask for the image found

   print(f"matched: {len(images)} | skipped: {len(skipped)}")
   return images , masks

# get the matched image and mask lists
images, masks = folder_path(image_path, mask_path)


#shuffle the images and maks in datafolder
pairs = list(zip(images, masks))
random.Random(42).shuffle(pairs)
images, masks = map(list, zip(*pairs))


#geometric changes are applied to both image and mask
train_transforms = A.Compose([
    A.Resize(height= 256 , width =256),
    A.Rotate(limit = 35 , p=0.5),
    A.HorizontalFlip(p=0.4),
    A.Normalize(mean= (0.485 , 0.456 , 0.406) , std = (0.229 , 0.224 , 0.225) , max_pixel_value = 255.0),# scaling the pixel value 0-1 plus standardized
    ToTensorV2() # converting the numpy values into tensor

    ])


valid_transforms = A.Compose([
    A.Resize(height = 256 , width = 256),
   A.Normalize(mean= (0.485 , 0.456 , 0.406) , std = (0.229 , 0.224 , 0.225) , max_pixel_value = 255.0),
    ToTensorV2()
])


class Custom(Dataset):
    def __init__(self , image_path , mask_path , transforms= None, split_type= "train"):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transforms = transforms

        #splitting folder in ratio : 70(train)/15(valid)/15(test)
        train_ratio = int(len(self.image_path)  * 0.70)
        valid_ratio = int(len(self.image_path) * 0.15)
        valid_end = train_ratio + valid_ratio

        if split_type == "train":
            self.images = self.image_path[:train_ratio]
            self.masks = self.mask_path[:train_ratio]
            print(f"the len of train folder is {len(self.images)} and {len(self.masks)}")
        elif split_type == "valid":
            self.images = self.image_path[train_ratio:valid_end]
            self.masks = self.mask_path[train_ratio:valid_end]
            print(f"the len of valid folder is {len(self.images)} and {len(self.masks)}")
        else:
            self.images = self.image_path[valid_end:]
            self.masks = self.mask_path[valid_end:]
            print(f"the len of test folder is {len(self.images)} and {len(self.masks)}")


    def __len__(self):
       return len(self.images)

    def __getitem__(self, idx):
       image_path = self.images[idx]
       image_rgb = cv2.imread(image_path)
       image = cv2.cvtColor(image_rgb , cv2.COLOR_BGR2RGB) #convert the format BGR --> RGB

       mask_path = self.masks[idx]
       mask_binary = cv2.imread(mask_path , cv2.IMREAD_GRAYSCALE)  #read the mask as one channel (0-255), then make it binary 0/1
       mask = np.where(mask_binary > 0 , 1, 0).astype(np.float32)

       if self.transforms:
           augmentations = self.transforms(image = image , mask= mask)
           image = augmentations["image"]
           mask = augmentations["mask"].unsqueeze(0)
       return image , mask




train_dataset = Custom(images, masks , train_transforms, "train")
valid_dataset = Custom(images, masks, valid_transforms , "valid")
test_dataset = Custom(images, masks , valid_transforms , "test")

train_loader = DataLoader(train_dataset , batch_size = 8 , shuffle = True , num_workers = 2 ,  pin_memory = True )
valid_loader = DataLoader(valid_dataset , batch_size= 8)
test_loader = DataLoader(test_dataset , batch_size = 2)
