
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.search import search_similar_images

st.set_page_config(
    page_title="Image Similarity Search",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Image Similarity Search")
st.write("Upload a painting to find the five most visually similar images.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Your query image")
    st.image(query_image, width=300)

    if st.button("Find Similar Images"):
        with st.spinner("Finding similar paintings..."):
            results = search_similar_images(query_image, top_k=5)

        st.subheader("Top 5 Similar Images")

        columns = st.columns(5)

        for column, result in zip(columns, results):
            with column:
                st.image(
                    result["image_path"],
                    width="stretch"
                )
                st.caption(Path(result["image_path"]).name)
                st.write(
                    f"Similarity: {result['similarity_score']:.4f}"
                )