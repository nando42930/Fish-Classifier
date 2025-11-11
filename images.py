from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from pathlib import Path


def load_and_preprocess_image(img_path: Path, target_size=(224, 224)) -> np.ndarray:
    """Load and preprocess an image for MobileNetV2."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)
