
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_model, get_embedding

EMBEDDINGS_FILE = PROJECT_ROOT / "embeddings" / "image_embeddings.npz"


def search_similar_images(query_image, top_k=5):
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run python src/embed.py first."
        )

    # Load saved embeddings and relative image paths
    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        paths = data["paths"]

    # Load the AI model and generate the query embedding
    model, preprocess, device = load_model()
    query_vector = get_embedding(
        query_image, model, preprocess, device
    )

    # Calculate cosine similarity (embeddings are normalized)
    scores = embeddings @ query_vector

    # Resolve saved paths relative to the project root
    resolved_paths = [
        (PROJECT_ROOT / Path(path)).resolve()
        for path in paths
    ]

    # Exclude the query image itself when it is a local file
    if isinstance(query_image, (str, Path)):
        query_path = Path(query_image).resolve()
        scores = scores.copy()

        for i, path in enumerate(resolved_paths):
            if path == query_path:
                scores[i] = -1.0

    # Select the top matching images
    top_k = min(max(0, top_k), len(paths))
    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "image_path": str(resolved_paths[i]),
            "similarity_score": float(scores[i])
        }
        for i in indices
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to query image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = search_similar_images(args.image, args.top_k)

    print("\nMost similar images:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['image_path']} | "
            f"Similarity: {result['similarity_score']:.4f}"
        )
