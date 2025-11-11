from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "data" / "species_dataset"
SAMPLE_DIR = BASE_DIR / "data" / "sample"

# Embedding cache file
EMBEDDINGS_PATH = BASE_DIR / "species_embeddings.npy"

# Image settings
IMAGE_SIZE = (224, 224)
