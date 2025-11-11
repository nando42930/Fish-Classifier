import numpy as np
from tqdm import tqdm
from pathlib import Path
from images import load_and_preprocess_image


def compute_species_embeddings(base_model, dataset_dir: Path, target_size=(224, 224)) -> dict:
    """Compute a mean embedding for each species in the dataset."""
    species_embeddings = {}

    for species_dir in tqdm(list(dataset_dir.iterdir()), desc="Extracting embeddings"):
        if not species_dir.is_dir():
            continue

        embeddings = []
        for img_path in species_dir.rglob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                img_array = load_and_preprocess_image(img_path, target_size)
                embedding = base_model.predict(img_array, verbose=0)
                embeddings.append(embedding)
            except Exception as e:
                print(f"⚠️ Skipping {img_path.name}: {e}")

        if embeddings:
            species_embeddings[species_dir.name] = np.mean(embeddings, axis=0)

    return species_embeddings


def save_embeddings(species_embeddings: dict, save_path: Path):
    """Save embeddings dictionary to a .npy file."""
    np.save(save_path, species_embeddings)
    print(f"💾 Saved embeddings to {save_path}")


def load_embeddings(load_path: Path) -> dict:
    """Load embeddings from .npy if available."""
    if load_path.exists():
        data = np.load(load_path, allow_pickle=True).item()
        print(f"📂 Loaded cached embeddings from {load_path}")
        return data
    return {}
