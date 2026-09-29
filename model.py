
import torch
import torch.nn as nn
import torchvision.models as models

"""
Lip segmentation model: MobileNetV3-Small encoder (ImageNet pretrained) + custom U-Net style decoder.

"""


class DecoderBlock(nn.Module):
    def __init__(self , in_channels , skip_channels , out_channels):
       super().__init__()
       self.up = nn.ConvTranspose2d(in_channels , out_channels , kernel_size = 2 , stride =2) # upsample the feature map 2x
       #two 3x3 convs to refine the merged features
       self.cnn = nn.Sequential(
           nn.Conv2d(out_channels + skip_channels ,out_channels , 3 , padding=1),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace= True),

           nn.Conv2d(out_channels , out_channels , 3 , padding=1),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace= True),
       )

    def forward(self , x , skip):
      x = self.up(x)
      if x.shape[-2:] != skip.shape[-2:]:
        x = nn.functional.interpolate(x , size=skip.shape[-2:] , mode= "bilinear" , align_corners = False)
      x = torch.cat([x, skip] , dim=1) # skip connection: join encoder features along channels
      return self.cnn(x)




class MobileNet(nn.Module):
    def __init__(self , num_classes =1 , pretrained = True):
        super().__init__()
         # MobileNetV3-Small: keep only the feature extractor (encoder), drop the classifier
        backbone = models.mobilenet_v3_small(
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        ).features

         # encoder stages, each saves a feature map for the decoder's skip connections
        self.enc0 = backbone[0:1]
        self.enc1 = backbone[1:2]
        self.enc2 = backbone[2:4]
        self.enc3 = backbone[4:9]
        self.enc4 = backbone[9:12]

        # decoder: (input_channels , skip_channels , output_channels)
        self.dec4 = DecoderBlock(96 , 48 , 128)
        self.dec3 = DecoderBlock(128 , 24 , 64)
        self.dec2 = DecoderBlock(64 , 16 , 32)
        self.dec1 = DecoderBlock(32 , 16 , 16)

        self.final_up = nn.ConvTranspose2d(16 , 16 , kernel_size = 2, stride = 2) # last 2x upsample, back to full size
        self.head = nn.Conv2d(16 , num_classes , kernel_size =1) ## 1x1 conv, gives one logit per pixel

    def forward(self , x):
      s0 = self.enc0(x)
      s1 = self.enc1(s0)
      s2 = self.enc2(s1)
      s3 = self.enc3(s2)
      s4 = self.enc4(s3)

      d4 = self.dec4(s4 , s3)
      d3 = self.dec3(d4 , s2)
      d2 = self.dec2(d3 , s1)
      d1 = self.dec1(d2 , s0)
      out = self.final_up(d1)
      return self.head(out)



