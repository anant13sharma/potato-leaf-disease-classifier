import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from tensorflow.keras.preprocessing import image
from fastapi import File

# Initialize app
app = FastAPI()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
# Load model
model = tf.keras.models.load_model("potatoes.h5", compile=False)
class_names = ['Early Blight', 'Late Blight', 'Healthy']


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    base_dir = "static/samples"
    class_folders = [
        "Potato___Early_blight",
        "Potato___healthy",
        "Potato___Late_blight"
    ]

    samples = []
    for folder in class_folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                chosen = np.random.choice(images)
                img_path = f"{base_dir}/{folder}/{chosen}"
                category = folder.replace("Potato___", "")
                samples.append((category, img_path))

    return templates.TemplateResponse("index.html", {"request": request, "samples": samples})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)

    img = image.load_img("temp.jpg", target_size=(256, 256))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)[0]
    label = class_names[np.argmax(prediction)]

    return {"label": label}

@app.post("/predict-sample")
async def predict_sample(request: Request, path: str = Form(...)):
    path = path.lstrip("/")
    img = image.load_img(path, target_size=(256, 256))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)[0]
    label = class_names[np.argmax(prediction)]

    return templates.TemplateResponse("result.html", {"request": request, "label": label})
