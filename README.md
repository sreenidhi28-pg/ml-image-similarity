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
- Supports local execution and Docker-based deployment.

## Technologies Used

- Python
- OpenCLIP
- PyTorch
- NumPy
- Pillow
- Streamlit
- tqdm
- Docker

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
├── Dockerfile
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/sreenidhi28-pg/ml-image-similarity.git
cd ml-image-similarity
```

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

This project uses the WikiArt image dataset, organized by art movement and artist.

The current dataset contains **4,875 images from the Abstract Expressionism collection**.

Dataset source: [WikiArt Image Dataset](https://github.com/asahi417/wikiart-image-dataset)

Download the required artwork images and extract them into:

```text
dataset/processed_wikiart/
```

The application searches recursively for supported image files, including JPG, JPEG, PNG, WEBP, and BMP.

> **Note:** The dataset and generated embedding file may be excluded from GitHub because of their size. Make sure they are available locally before running the application.

## Usage

### 1. Generate image embeddings

Run this command from the project root:

```powershell
python src/embed.py
```

The script processes the dataset images and saves their embeddings and relative file paths to:

```text
embeddings/image_embeddings.npz
```

Run this command again whenever you change the indexed dataset.

### 2. Search for similar images using the command line

Provide the path to an existing image in the dataset:

```powershell
python src/search.py --image "dataset/processed_wikiart/abstract_expressionism/aaron-siskind/acolman-1-1955.jpg"
```

The program returns up to five matching images with similarity scores. If the query image belongs to the indexed dataset, that exact image is excluded from the results.

### 3. Run the Streamlit application

```powershell
python -m streamlit run demo/app.py
```

Open the local URL displayed in the terminal. Upload a query image and click **Find Similar Images** to view the matching images and their scores.

## Docker Usage

Docker allows the application to run inside a container with its required dependencies.

### 1. Build the Docker image

From the project root, run:

```powershell
docker build -t ml-image-similarity .
```

### 2. Start the container

The following command maps port 8502 on your computer to port 8501 inside the container:

```powershell
docker run -d -p 8502:8501 --name image-similarity-app ml-image-similarity
```

### 3. Open the application

Visit:

```text
http://localhost:8502
```

Upload an image and verify that the application displays five similar images and their similarity scores.

### 4. Check the container

```powershell
docker ps
```

### 5. Stop the container

```powershell
docker stop image-similarity-app
```

To start the existing container again:

```powershell
docker start image-similarity-app
```

If the container name is already in use, reuse the existing container rather than creating another one with the same name.

## How It Works

1. A pretrained OpenCLIP model loads each dataset image.
2. Each image is converted into a numerical embedding.
3. The embedding is normalized and saved alongside the image's relative path.
4. The uploaded query image is converted into an embedding using the same model.
5. Cosine similarity is calculated between the query embedding and the saved embeddings.
6. The highest-scoring matches are returned and displayed in the application.

## Similarity Score

A higher cosine similarity score generally indicates that two image embeddings are more similar.

The similarity score is not a percentage and does not guarantee that the images depict the same subject. Results reflect visual similarity in the embedding space learned by the pretrained model.

## Current Project Status

- Successfully generated embeddings for 4,875 images.
- Successfully tested command-line similarity search.
- Successfully tested the Streamlit web application.
- Successfully built the Docker image.
- Successfully launched the application inside a Docker container.
- Successfully tested three different query images, each returning five similar images with similarity scores.

## Future Improvements

- Expand the dataset with additional WikiArt art movements.
- Evaluate retrieval quality using a larger set of query images.
- Improve search performance for larger datasets.
- Add progress indicators and improved error handling.
- Deploy the application online for public access.

