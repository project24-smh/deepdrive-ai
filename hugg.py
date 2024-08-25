import os
import tempfile
import time
import streamlit as st
from gradio_client import Client
import requests
from PIL import Image
import io
import base64

# Initialize the Gradio clients
upscale_client = Client("gokaygokay/TileUpscalerV2")
new_model_client = Client("prithivMLmods/FLUX.1-SIM")

# Hugging Face API configuration
IMAGE_GEN_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
IMAGE_GEN_HEADERS = {"Authorization": "Bearer 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"}

# Helper functions
def query(payload, url, headers):
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying API: {e}")
        return None

def generate_image(prompt):
    try:
        image_bytes = query({"inputs": prompt}, IMAGE_GEN_API_URL, IMAGE_GEN_HEADERS)
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Error generating image: {e}")
    return None

def upscale_image(image_file_path, upscale_to, steps, noise_level, fidelity, seed, guidance_scale, sampler):
    try:
        result = upscale_client.predict(
            image_file_path,
            upscale_to,
            steps,
            noise_level,
            fidelity,
            seed,
            guidance_scale,
            sampler,
            api_name="/wrapper"
        )
        return result
    except Exception as e:
        st.error(f"Error upscaling image: {e}")
        return None

def new_model_inference(prompt, seed, randomize_seed, wallpaper_size, num_inference_steps, style_name):
    try:
        result = new_model_client.predict(
            prompt,
            seed,
            randomize_seed,
            wallpaper_size,
            num_inference_steps,
            style_name,
            api_name="/infer"
        )
        return result
    except Exception as e:
        st.error(f"Error with new model inference: {e}")
        return None

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Initialize session state
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "face_swap_result" not in st.session_state:
    st.session_state.face_swap_result = None
if "upscaled_image" not in st.session_state:
    st.session_state.upscaled_image = None
if "new_model_result" not in st.session_state:
    st.session_state.new_model_result = None

# Main app layout
st.sidebar.title("Select an Option")
app_mode = st.sidebar.radio(
    "Choose an option",
    ["Gen.Schnell", "Flux1 Ultimate", "Int. Dalle-3/F", "Face Swap", "Image Upscaler"]
)

if app_mode == "Gen.Schnell":
    st.title("Generate Image from Prompt")

    prompt = st.text_input("Enter a prompt for the image:")

    if st.button("Generate Image"):
        if prompt:
            with st.spinner('Generating image...'):
                start_time = time.time()
                st.session_state.generated_image = generate_image(prompt)
                processing_time = round(time.time() - start_time, 2)
                st.write(f"Processing Time: {processing_time} seconds")
        else:
            st.error("Please enter a prompt.")

    if st.session_state.generated_image:
        st.image(st.session_state.generated_image, caption="Generated Image", use_column_width=True)
        buffer = io.BytesIO()
        st.session_state.generated_image.save(buffer, format="JPEG")
        buffer.seek(0)

        st.download_button(
            label="Download Generated Image",
            data=buffer,
            file_name="generated_image.jpg",
            mime="image/jpeg"
        )

elif app_mode == "Face Swap":
    st.title("Face Swap App")

    source_image = st.file_uploader("Choose a source image...", type=["jpg", "jpeg"])
    target_image = st.file_uploader("Choose a target image...", type=["jpg", "jpeg"])

    if st.button("Swap Faces"):
        if source_image and target_image:
            source_img = Image.open(source_image)
            target_img = Image.open(target_image)

            source_b64 = image_to_base64(source_img)
            target_b64 = image_to_base64(target_img)

            api_key = st.secrets["SEGMIND_API_KEY"]
            url = "https://api.segmind.com/v1/faceswap-v2"
            data = {
                "source_img": source_b64,
                "target_img": target_b64,
                "input_faces_index": 0,
                "source_faces_index": 0,
                "face_restore": "codeformer-v0.1.0.pth",
                "base64": False
            }

            headers = {
                'x-api-key': api_key,
                'Content-Type': 'application/json'
            }

            try:
                response = requests.post(url, json=data, headers=headers)
                response.raise_for_status()
                image_bytes = io.BytesIO(response.content)
                st.session_state.face_swap_result = Image.open(image_bytes)
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please upload both source and target images.")

    if st.session_state.face_swap_result:
        st.image(st.session_state.face_swap_result, caption="Face Swapped Image", use_column_width=True)
        buffer = io.BytesIO()
        st.session_state.face_swap_result.save(buffer, format="JPEG")
        buffer.seek(0)

        st.download_button(
            label="Download Face Swapped Image",
            data=buffer,
            file_name="face_swapped_image.jpg",
            mime="image/jpeg"
        )

elif app_mode == "Int. Dalle-3/F":
    st.title("Dalle-3 Integration")

    st.markdown(
        """
        <iframe src="https://nymbo-flux-1-dev-serverless.hf.space/" 
        width="100%" height="800" frameborder="0" scrolling="auto"></iframe>
        """,
        unsafe_allow_html=True
    )

elif app_mode == "Image Upscaler":
    st.title("Image Upscaler")

    uploaded_file = st.file_uploader("Upload an image for upscaling", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

    advanced_options = st.expander("Advanced Options")
    with advanced_options:
        upscale_to = st.selectbox("Select the upscale resolution", [1024, 2048, 4096])
        steps = st.slider("Select the number of steps", 1, 100, 20)
        noise_level = st.slider("Select the noise level", 0.0, 1.0, 0.2, format="%.2f")
        fidelity = st.slider("Select the fidelity level", 0.0, 1.0, 0.0, format="%.2f")
        seed = st.slider("Select the seed value", 0, 100, 6)
        guidance_scale = st.slider("Select the guidance scale", 0.0, 1.0, 0.75, format="%.2f")
        sampler = st.selectbox("Select the sampler type", ["DDIM", "PNDMS", "Heun"])

    if st.button("Upscale Image"):
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            with st.spinner('Upscaling image...'):
                start_time = time.time()
                result = upscale_image(temp_file_path, upscale_to, steps, noise_level, fidelity, seed, guidance_scale, sampler)
                processing_time = round(time.time() - start_time, 2)
                st.write(f"Processing Time: {processing_time} seconds")

                if result:
                    upscaled_image = Image.open(result)
                    st.session_state.upscaled_image = upscaled_image
                else:
                    st.error("Failed to upscale the image.")

            os.unlink(temp_file_path)

    if st.session_state.upscaled_image:
        st.image(st.session_state.upscaled_image, caption="Upscaled Image", use_column_width=True)
        buffer = io.BytesIO()
        st.session_state.upscaled_image.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            label="Download Upscaled Image",
            data=buffer,
            file_name="upscaled_image.png",
            mime="image/png"
        )

elif app_mode == "Flux Ultimate":
    st.title("Flux Ultimate Integration")

    prompt = st.text_input("Enter prompt for Flux:")
    
    advanced_options = st.expander("Advanced Options")
    with advanced_options:
        seed = st.slider("Select the seed value", 0, 100, 0)
        randomize_seed = st.checkbox("Randomize Seed", value=True)
        wallpaper_size = st.selectbox("Select wallpaper size", ["Default (1024x1024)", "Small (512x512)", "Large (2048x2048)"])
        num_inference_steps = st.slider("Select number of inference steps", 1, 100, 4)
        style_name = st.selectbox("Select style name", ["Style Zero", "Style One", "Style Two"])

    if st.button("Run Flux Ultimate"):
        if prompt:
            with st.spinner('Processing...'):
                start_time = time.time()
                result = new_model_inference(prompt, seed, randomize_seed, wallpaper_size, num_inference_steps, style_name)
                processing_time = round(time.time() - start_time, 2)
                st.write(f"Processing Time: {processing_time} seconds")

                if result:
                    image_data = result[0]  # Adjust this based on actual result structure

                    if isinstance(image_data, str) and (image_data.endswith('.png') or image_data.endswith('.jpg')):
                        image = Image.open(image_data)
                    elif isinstance(image_data, (bytes, bytearray)):
                        image = Image.open(io.BytesIO(image_data))
                    elif isinstance(image_data, Image.Image):
                        image = image_data
                    else:
                        st.error("Unexpected result format.")
                        image = None
                    
                    if image:
                        st.image(image, caption="Generated Image", use_column_width=True)
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG")
                        buffer.seek(0)

                        st.download_button(
                            label="Download Generated Image",
                            data=buffer,
                            file_name="new_model_generated_image.jpeg",
                            mime="image/jpeg"
                        )
                    else:
                        st.error("Failed to process the image.")
                else:
                    st.error("Failed to generate image.")
        else:
            st.error("Please enter a prompt.")
