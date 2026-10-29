
import torch
import open_clip
from PIL import Image


def load_model():
    # Use GPU if available; otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained OpenCLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    model = model.to(device)
    model.eval()

    return model, preprocess, device


def get_embedding(image, model, preprocess, device):
    # Accept an image path or an already opened image
    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGB")

    # Prepare image for the AI model
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Convert image into a numerical vector
    with torch.no_grad():
        features = model.encode_image(image_tensor)

        # Normalize the vector for similarity comparison
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()[0]
