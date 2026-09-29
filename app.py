from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import uuid
import os
from main import VirtualTryOn

# create the FastAPI app instance
app = FastAPI()
os.makedirs("outputs", exist_ok=True)

# serve everything inside the static/ folder (index.html, sample images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

#landing page
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.post("/tryon/image")
async def tryon_image(
    file: UploadFile = File(...), # the uploaded image file
    r: int = Form(204), 
    g: int = Form(124),
    b: int = Form(34),
    strength: float = Form(0.6),
    gloss: int = Form(30),
):
    tryon = VirtualTryOn(color=(r, g, b), strength=strength, gloss=gloss)

    contents = await file.read() # read the uploaded file's raw bytes
    np_arr = np.frombuffer(contents, np.uint8)  # convert raw bytes into a numpy array
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # decode bytes into an actual image (BGR format)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mask = tryon.predict_mask(rgb)
    result = tryon.apply(rgb, mask)

    out_path = f"outputs/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return FileResponse(out_path, media_type="image/jpeg")  # send the result image back to the client


@app.post("/tryon/video")
async def tryon_video(
    file: UploadFile = File(...),
    r: int = Form(204),
    g: int = Form(124),
    b: int = Form(34),
    strength: float = Form(0.6),
    gloss: int = Form(30),
):
    tryon = VirtualTryOn(color=(r, g, b), strength=strength, gloss=gloss)

    in_path = f"outputs/{uuid.uuid4().hex}_in.mp4"
    out_path = f"outputs/{uuid.uuid4().hex}_out.mp4"
    with open(in_path, "wb") as f:
        f.write(await file.read())

    tryon.process_video(in_path, out_path)
    return FileResponse(out_path, media_type="video/mp4")