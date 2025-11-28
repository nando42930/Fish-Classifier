import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from models.mobilenet import build_feature_extractor
from utils.similarity import classify_image
from config import IMAGE_SIZE, EMBEDDINGS_PATH
import images

app = FastAPI()

# Build model once at startup
base_model = build_feature_extractor()

# Load embeddings once at startup
species_embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded bytes
    contents = await file.read()

    # Convert bytes to PIL Image
    img = Image.open(BytesIO(contents)).convert("RGB")

    # Preprocess image for MobileNetV2
    img_array = images.load_and_preprocess_image(img, target_size=IMAGE_SIZE)

    # Run your classifier
    pred, conf = classify_image(base_model, species_embeddings, img_array, IMAGE_SIZE)

    # Format result for display
    formatted_result = f"Prediction: {pred}\nConfidence: {conf:.3f}"

    return {"result": formatted_result}
