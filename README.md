# Fish Species Classifier Using Deep Feature Embeddings

![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)


## Overview
This project implements an automated fish species classifier using deep learning feature embeddings. By leveraging MobileNetV2 as a pretrained feature extractor, it can classify images of fish into their correct species based on a dataset organized into species folders.


### Dataset Overview
1. **Total images:** 3963
2. **Total species:** 471
3. **Image capture conditions:**
   1. **Controlled:** Fish specimens with fins spread, photographed against a constant background with controlled illumination.
   2. **In-situ:** Underwater images of fish in their natural habitat, with uncontrolled background and lighting.
   3. **Out-of-the-water:** Fish specimens photographed out of water, with varying backgrounds and limited control over illumination.

This diverse dataset ensures that the classifier can generalize across multiple imaging conditions, making it suitable for both laboratory and real-world applications.

The approach is few-shot capable, extensible and does not require retraining a full classifier every time a new species is added.


## Project Structure
```
fish_classifier/
├── main.py              # Entry point
├── config.py            # Paths and constants
├── models/mobilenet.py  # Pretrained feature extractor
├── utils/
│   ├── images.py        # Image loading and preprocessing
│   ├── embeddings.py    # Embedding computation and caching
│   └── similarity.py    # Cosine similarity and classification
└── data/
    ├── species_dataset/ # Reference images for each species
    └── sample/          # Test images to classify
```


### Dataset Organization
The dataset should be organized as:
```
data/species_dataset/  
├── aprion_virescens/  
│   ├── aprion_virescens_1.jpg  
│   ├── aprion_virescens_2.jpg  
├── lutjanus_russellii/  
│   ├── lutjanus_russellii_1.jpg  
│   ├── lutjanus_russellii_2.jpg
```

* Each folder represents a species.
* Supported image formats: `.jpg`, `.jpeg`, `.png`.
* Subfolders are automatically processed when extracting embeddings.


## Installation
### Clone the repository
```bash
git clone https://github.com/yourusername/fish_classifier.git
cd fish_classifier
```


### Install dependencies (Python 3.8+ recommended)
```bash
pip install tensorflow numpy tqdm
```


## How It Works
### Feature Extraction
* MobileNetV2 is used without the classification head (`include_top=False`) and global average pooling (`pooling='avg'`).
* Each fish image is resized to 224×224 and preprocessed using MobileNetV2’s standard normalization.


### Embedding Computation
For each species folder:
1. All images are loaded and preprocessed.
2. Each image is passed through MobileNetV2 to generate a feature embedding.
3. The mean embedding for the species is calculated.

> Embeddings are cached to `species_embeddings.npy` to avoid recomputation.


### Classification
To classify a new image:
1. Preprocess the image and extract its embedding.
2. Compute cosine similarity with all species embeddings.
3. Return the species with the highest similarity:
```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Example usage
similarity = cosine_similarity(vec1, vec2)
```


## Usage
Classify a new image: 
```bash
python main.py
```

Example output:
> 🐟 Predicted species: aprion_virescens (similarity = 0.912)

This indicates the input image is most likely Aprion Virescens.


## Features
* ✅ Few-shot capable (works with only a few images per species)
* ✅ Supports multiple image formats (`.jpg`, `.jpeg`, `.png`)
* ✅ Fast classification via embedding caching
* ✅ Modular and extensible design
* ✅ Confidence scores via cosine similarity


## Future Improvements
* Fine-tuning the backbone on fish-specific datasets
* Using larger models like EfficientNet for better accuracy
* Deploying as a web API for interactive classification
* Adding a command-line interface (CLI) to classify arbitrary images


## References
- [Fish Species Dataset](https://www.kaggle.com/datasets/sripaadsrinivasan/fish-species-image-data)
- [Local Inter-Session Variability Modelling for Object Classification Paper](https://eprints.qut.edu.au/67786/)
