from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
import base64
import cv2
import io
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load your model and define classes
loaded_model = tf.keras.models.load_model('/workspace/model/model.h5', compile=False)
classes = {0: ('ca', 'colon adenocarcinoma'), 1: ('cb', 'colon benign'), 2: ('lac', 'lung adenocarcinoma'), 3: ('lb', 'lung benign'),
           4: ('lscc', 'lung squamous cell carcinoma')}  # Define your classes here

@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("report.html", {"request": request})

@app.post("/report")
async def report(request: Request, file: UploadFile = File(...)):
    s_img = await file.read()
    # Convert the bytes data to a NumPy array
    image = Image.open(io.BytesIO(s_img))

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = loaded_model.predict(img_array)
    max_prob = np.max(predictions)
    class_ind = np.argmax(predictions)
    class_name = classes[class_ind]

    result = {
        "prediction": class_name
    }
    print("Prediction:", class_name)
    return templates.TemplateResponse("report.html", {"request": request, "result": class_name})