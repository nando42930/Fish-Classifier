import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
from pathlib import Path
from tqdm import tqdm

# -------------------------------
# 1️⃣ Load pretrained model as feature extractor
# -------------------------------
base_model = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))

# -------------------------------
# 2️⃣ Extract embeddings for each species
# -------------------------------
reference_dir = Path("../data/species_dataset/")
species_embeddings = {}

# iterate over subfolders (each species)
for species_dir in reference_dir.iterdir():
    if not species_dir.is_dir():
        continue

    embeddings = []
    for img_path in species_dir.rglob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, 0)
            img_array = preprocess_input(img_array)

            embedding = base_model.predict(img_array, verbose=0)
            embeddings.append(embedding)
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

    if embeddings:
        species_embeddings[species_dir.name] = np.mean(embeddings, axis=0)

print(f"✅ Loaded embeddings for {len(species_embeddings)} species.")

# -------------------------------
# 3️⃣ Load new image to classify
# -------------------------------
new_img_path = Path("../data/sample/009.png")
img = image.load_img(new_img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, 0)
img_array = preprocess_input(img_array)
new_embedding = base_model.predict(img_array, verbose=0)

# -------------------------------
# 4️⃣ Compute cosine similarity
# -------------------------------
def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = {}
for species, emb in species_embeddings.items():
    similarities[species] = cosine_similarity(new_embedding, emb)

# -------------------------------
# 5️⃣ Pick the most similar species
# -------------------------------
predicted_species = max(similarities, key=similarities.get)
confidence = similarities[predicted_species]

print(f"\n🐟 Predicted species: {predicted_species} (similarity = {confidence:.3f})")
