
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Allow Python to find model.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_model, get_embedding

DATASET = PROJECT_ROOT / "dataset" / "processed_wikiart"
OUTPUT = PROJECT_ROOT / "embeddings" / "image_embeddings.npz"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    # Find all supported images in the dataset
    image_paths = sorted(
        path for path in DATASET.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print("No images found in:", DATASET)
        return

    print("Images found:", len(image_paths))
    print("Loading AI model...")

    model, preprocess, device = load_model()
    print("Device:", device)

    embeddings = []
    image_names = []

    # Generate an embedding for every image
    for path in tqdm(image_paths, desc="Processing images"):
        try:
            with Image.open(path) as image:
                vector = get_embedding(image, model, preprocess, device)

            embeddings.append(vector)

            # Save a relative path instead of a Windows absolute path
            image_names.append(
                path.relative_to(PROJECT_ROOT).as_posix()
            )

        except (OSError, ValueError, UnidentifiedImageError) as error:
            print(f"Skipping {path.name}: {error}")

    if not embeddings:
        print("No images could be processed.")
        return

    # Save embeddings and relative image paths
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        paths=np.asarray(image_names)
    )

    print(f"\nSuccess! Saved {len(embeddings)} image embeddings.")
    print("Output file:", OUTPUT)


if __name__ == "__main__":
    main()
