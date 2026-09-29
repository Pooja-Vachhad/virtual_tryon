# Virtual Lip Try-On

A virtual try-on app that detects the lip region on a face and applies lipstick color + gloss in real time, on photos and video.

## How it works

1. A U-Net style model (MobileNetV3-Small encoder + custom decoder) segments the lip region from the image.
2. The predicted mask is used to blend a chosen lipstick color onto the lips, with a gloss highlight effect.
3. Works on both single images and video (with frame-to-frame smoothing to avoid flicker).

## Model

- **Backbone:** MobileNetV3-Small (pretrained on ImageNet), chosen for fast, lightweight inference on edge devices.
- **Decoder:** Custom 3-layer U-Net style decoder with skip connections, for precise lip-edge segmentation.
- **Loss:** Dice + BCE loss, to handle the class imbalance between lips (small region) and background.

**Results:**
- Test Dice score: **0.9656**
- Test loss: **0.0379**

![Dice curve](assets/dice_curve.png)
![Loss curve](assets/loss_curve.png)

## Deployment

- The trained PyTorch model is exported to **ONNX** for faster, lighter inference (no PyTorch needed at serving time).
- Verified the ONNX export matches the original model's output (max difference ~0.0001).
- Wrapped in a **FastAPI** backend with endpoints for image and video processing.
- Simple web UI (color picker + strength/gloss sliders) to try it out.
- Containerized with **Docker** for consistent, portable deployment.

## Tech Stack
PyTorch • ONNX Runtime • FastAPI • OpenCV • Docker

## Limitations
- Trained for a limited number of epochs due to GPU constraints.
- Accuracy drops on some skin tones and lighting conditions not well represented in training data — a direction for future improvement.

## Demo
[Add your demo video link or screenshots here]

