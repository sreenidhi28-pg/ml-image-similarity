
# Image Similarity Search

## Project Overview

Image Similarity Search is a visual search application that finds the five most visually similar images to a query image. It uses a pretrained OpenCLIP model to convert images into numerical embeddings and ranks them using cosine similarity.

## Features

- Uses a pretrained OpenCLIP model.
- Generates embeddings for dataset images.
- Finds the top 5 visually similar images.
- Displays similarity scores for matching images.
- Provides an interactive Streamlit web interface.
- Supports searching through images stored in nested folders.

## Technologies Used

- Python
- OpenCLIP
- PyTorch
- NumPy
- Pillow
- Streamlit
- tqdm

## Project Structure

```text
ml-image-similarity/
├── src/
│   ├── model.py
│   ├── embed.py
│   └── search.py
├── dataset/
│   └── processed_wikiart/
├── embeddings/
│   └── image_embeddings.npz
├── demo/
│   └── app.py
├── requirements.txt
├── README.md
└── Dockerfile
```

## Installation

### 1. Clone or download the project

Open the project folder in a terminal.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Dataset

This project uses the WikiArt General dataset, organized by art movement and artist.

The current dataset contains 4,875 images from the Abstract Expressionism collection.

Dataset source: [WikiART Image Dataset](https://github.com/asahi417/wikiart-image-dataset)

Extract the downloaded artwork images into:

```text
dataset/processed_wikiart/
```

The application searches for supported image files recursively within this directory.

## Usage

### 1. Generate image embeddings

Run this command from the project root:

```powershell
python src/embed.py
```

The script processes supported images and saves their embeddings and file paths to:

```text
embeddings/image_embeddings.npz
```

Running this command again regenerates the embedding file.

### 2. Search for similar images

Provide the path to an existing image in the dataset:

```powershell
python src/search.py --image "dataset/processed_wikiart/abstract_expressionism/aaron-siskind/acolman-1-1955.jpg"
```

The program returns up to five matching images with similarity scores. If the query image is part of the indexed dataset, that exact file is excluded from the results.

### 3. Run the Streamlit application

```powershell
python -m streamlit run demo/app.py
```

Open the local URL displayed in the terminal. Upload an image and click **Find Similar Images** to view the matching paintings and their scores.

## How It Works

1. A pretrained OpenCLIP model loads each dataset image.
2. The image is converted into a numerical embedding.
3. The embedding is normalized and saved alongside the image path.
4. The query image is converted into an embedding using the same model.
5. Cosine similarity is calculated between the query vector and saved embeddings.
6. The highest-scoring matches are returned.

## Similarity Score

A higher cosine similarity score generally indicates that two image embeddings are more similar. The score is not a percentage and does not guarantee that the images depict the same subject.

## Current Status

- Successfully generated embeddings for 4,875 images.
- Successfully tested command-line similarity search.
- Successfully tested the Streamlit web application.
- Docker configuration has been created but still requires testing.

The current dataset covers Abstract Expressionism. Additional art movements can be added to expand the searchable collection.

## Future Improvements

- Expand the dataset with additional WikiArt art movements.
- Test with more query images and evaluate retrieval quality.
- Improve search performance for larger datasets.
- Test and validate Docker deployment.
- Improve handling of missing files and invalid query images.
