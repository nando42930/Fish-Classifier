import os
from pathlib import Path
from config import DATASET_DIR, SAMPLE_DIR, EMBEDDINGS_PATH, IMAGE_SIZE
from models.mobilenet import build_feature_extractor
from utils.embeddings import compute_species_embeddings, save_embeddings, load_embeddings
from utils.similarity import classify_image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TensorFlow logs


def main():
    print("🔧 Loading MobileNetV2 model...")
    base_model = build_feature_extractor()

    # Try to load cached embeddings
    species_embeddings = load_embeddings(EMBEDDINGS_PATH)

    # If not cached, compute and save
    if not species_embeddings:
        print("🧠 Computing new embeddings...")
        species_embeddings = compute_species_embeddings(base_model, DATASET_DIR, IMAGE_SIZE)
        save_embeddings(species_embeddings, EMBEDDINGS_PATH)

    # Classify a test image
    test_image = SAMPLE_DIR / "003.png"
    print(f"\n🔍 Classifying image: {test_image}")
    predicted_species, confidence = classify_image(base_model, species_embeddings, test_image, IMAGE_SIZE)

    print(f"\n🐟 Predicted species: {predicted_species} (similarity = {confidence:.3f})")


if __name__ == "__main__":
    main()
