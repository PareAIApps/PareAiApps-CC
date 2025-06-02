from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
from config import MODEL_PATH, IMAGE_SIZE

# Load model saat modul diimpor
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    model = None
    print(f"Model failed to load: {e}")

def get_model():
    return model

def preprocess_image(file):
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
