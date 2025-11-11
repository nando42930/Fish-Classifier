import numpy as np
from images import load_and_preprocess_image


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def classify_image(base_model, species_embeddings: dict, image_path, target_size=(224, 224)):
    """Predict the closest species based on cosine similarity."""
    img_array = load_and_preprocess_image(image_path, target_size)
    new_embedding = base_model.predict(img_array, verbose=0)

    similarities = {
        species: cosine_similarity(new_embedding, emb)
        for species, emb in species_embeddings.items()
    }

    predicted_species = max(similarities, key=similarities.get)
    confidence = similarities[predicted_species]
    return predicted_species, confidence
