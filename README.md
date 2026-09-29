# Virtual Lip Try-On

A virtual try-on app that detects the lip region on a face and applies lipstick color + gloss in real time, on photos and video.

<img width="1000" height="750" alt="Image" src="https://github.com/user-attachments/assets/1799ccc4-82cc-4a2f-a559-e4b91ad81c51" />

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


<img width="590" height="390" alt="Image" src="https://github.com/user-attachments/assets/b32d1cc1-30d9-4de9-88db-6a09f1a0e274" />

<img width="590" height="390" alt="Image" src="https://github.com/user-attachments/assets/1c9311da-9841-4602-a1a5-811fff9f6e3e" />

<img width="990" height="1928" alt="Image" src="https://github.com/user-attachments/assets/e905e3e9-fdf2-4bb5-9841-45e42f4626ce" />


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
https://github.com/user-attachments/assets/f7f64588-ae71-4fcf-aa4b-306e4f084d33

