import onnxruntime as ort
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

valid_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2()
])

# load the ONNX inference session
session = ort.InferenceSession("lip_seg.onnx")
# get the exact input/output names the model expects
INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


class VirtualTryOn:
    def __init__(self, color=(255, 120, 60), strength=0.6, gloss=50):
        self.color, self.strength, self.gloss = color, strength, gloss
        self.prev_mask = None

    def predict_mask(self, rgb):
        h, w = rgb.shape[:2]
        x = valid_transforms(image=rgb)["image"].unsqueeze(0).numpy().astype(np.float32)
        logits = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0][0, 0]
        logits = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR)
        return 1 / (1 + np.exp(-logits))

    def apply(self, rgb, mask):
        img = rgb.astype(np.float32)
        soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        alpha = (soft * self.strength)[..., None]
        color_layer = np.empty_like(img); color_layer[:] = self.color
        result = img * (1 - alpha) + color_layer * alpha

        binary = mask > 0.5
        if self.gloss > 0 and binary.any():
            gray = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            thr = np.percentile(gray[binary], 85)
            spot = ((gray > thr) & binary).astype(np.float32) * self.gloss
            result += cv2.GaussianBlur(spot, (9, 9), 0)[..., None]
        return np.clip(result, 0, 255).astype(np.uint8)

    def process_image(self, path, output_path):
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        out = self.apply(rgb, self.predict_mask(rgb))
        cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    def process_video(self, path, output_path):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = self.predict_mask(rgb)
           #applying if perv_maks doesn't exist if exist then blend it
            self.prev_mask = mask if self.prev_mask is None else 0.6 * self.prev_mask + 0.4 * mask
            result = self.apply(rgb, self.prev_mask)
            writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cap.release()
        writer.release()