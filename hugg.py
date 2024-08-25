import os
import io
import base64
import logging
import tempfile
import time
from datetime import datetime

import streamlit as st
import requests
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(page_title="Image Processing App", layout="wide")

# Initialize session state
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "upscaled_image" not in st.session_state:
    st.session_state.upscaled_image = None
if "face_swap_result" not in st.session_state:
    st.session_state.face_swap_result = None

# Helper functions
def log_error(error_message):
    logger.error(f"{datetime.now()}: {error_message}")
    st.error(error_message)

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_image(prompt):
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        log_error(f"Error generating image: {e}")
    return None

def upscale_image(image_file_path, upscale_factor=2):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/BiT-M-R50x1-SR"
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}"}
    
    try:
        with open(image_file_path, "rb") as f:
            response = requests.post(API_URL, headers=headers, data=f.read())
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        log_error(f"Error upscaling image: {e}")
    return None

def face_swap(source_image, target_image):
    API_URL = "https://api.segmind.com/v1/faceswap-v2"
    headers = {"x-api-key": st.secrets["SEGMIND_API_KEY"]}
    
    try:
        data = {
            "source_img": image_to_base64(source_image),
            "target_img": image_to_base64(target_image),
            "input_faces_index": 0,
            "source_faces_index": 0,
            "face_restore": "codeformer-v0.1.0.pth",
            "base64": False
        }
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        log_error(f"Error in face swap: {e}")
    return None

# Main app
def main():
    st.title("Image Processing App")

    # Sidebar for app mode selection
    app_mode = st.sidebar.selectbox(
        "Choose a mode",
        ["Image Generation", "Image Upscaling", "Face Swap"]
    )

    if app_mode == "Image Generation":
        st.header("Generate Image from Prompt")
        prompt = st.text_input("Enter a prompt for the image:")

        if st.button("Generate Image"):
            if prompt:
                with st.spinner('Generating image...'):
                    start_time = time.time()
                    st.session_state.generated_image = generate_image(prompt)
                    processing_time = round(time.time() - start_time, 2)
                    
                    if st.session_state.generated_image:
                        st.image(st.session_state.generated_image, caption="Generated Image", use_column_width=True)
                        st.write(f"Processing Time: {processing_time} seconds")
                        
                        # Provide download option
                        buf = io.BytesIO()
                        st.session_state.generated_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download Generated Image",
                            data=buf.getvalue(),
                            file_name="generated_image.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Failed to generate image. Please try again.")
            else:
                st.warning("Please enter a prompt.")

    elif app_mode == "Image Upscaling":
        st.header("Image Upscaling")
        uploaded_file = st.file_uploader("Choose an image to upscale", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)

            if st.button("Upscale Image"):
                with st.spinner('Upscaling image...'):
                    start_time = time.time()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                        image.save(temp_file, format="PNG")
                        temp_file_path = temp_file.name

                    st.session_state.upscaled_image = upscale_image(temp_file_path)
                    os.unlink(temp_file_path)  # Remove the temporary file
                    
                    processing_time = round(time.time() - start_time, 2)
                    
                    if st.session_state.upscaled_image:
                        st.image(st.session_state.upscaled_image, caption="Upscaled Image", use_column_width=True)
                        st.write(f"Processing Time: {processing_time} seconds")
                        
                        # Provide download option
                        buf = io.BytesIO()
                        st.session_state.upscaled_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download Upscaled Image",
                            data=buf.getvalue(),
                            file_name="upscaled_image.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Failed to upscale image. Please try again.")

    elif app_mode == "Face Swap":
        st.header("Face Swap")
        source_image = st.file_uploader("Choose a source image (face to use)", type=["png", "jpg", "jpeg"])
        target_image = st.file_uploader("Choose a target image (image to put the face on)", type=["png", "jpg", "jpeg"])

        if source_image and target_image:
            source_img = Image.open(source_image)
            target_img = Image.open(target_image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(source_img, caption="Source Image", use_column_width=True)
            with col2:
                st.image(target_img, caption="Target Image", use_column_width=True)

            if st.button("Swap Faces"):
                with st.spinner('Swapping faces...'):
                    start_time = time.time()
                    st.session_state.face_swap_result = face_swap(source_img, target_img)
                    processing_time = round(time.time() - start_time, 2)
                    
                    if st.session_state.face_swap_result:
                        st.image(st.session_state.face_swap_result, caption="Face Swap Result", use_column_width=True)
                        st.write(f"Processing Time: {processing_time} seconds")
                        
                        # Provide download option
                        buf = io.BytesIO()
                        st.session_state.face_swap_result.save(buf, format="PNG")
                        st.download_button(
                            label="Download Face Swap Result",
                            data=buf.getvalue(),
                            file_name="face_swap_result.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Face swap failed. Please try again with different images.")
        else:
            st.warning("Please upload both source and target images.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")
        st.error("An unexpected error occurred. Please try refreshing the page or contact support.")
