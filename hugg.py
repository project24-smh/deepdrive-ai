import os
import tempfile
import time
import streamlit as st
from gradio_client import Client, handle_file
import requests
from PIL import Image
import io
import base64

# Initialize the Gradio clients
@st.cache_resource
def get_upscale_client():
    return Client("gokaygokay/TileUpscalerV2")

@st.cache_resource
def get_new_model_client():
    return Client("prithivMLmods/FLUX.1-SIM")

# Hugging Face API URL and headers
IMAGE_GEN_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Use environment variable for sensitive information
IMAGE_GEN_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

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
    client = get_upscale_client()
    try:
        result = client.predict(
            param_0=handle_file(image_file_path),
            param_1=upscale_to,
            param_2=steps,
            param_3=noise_level,
            param_4=fidelity,
            param_5=seed,
            param_6=guidance_scale,
            param_7=sampler,
            api_name="/wrapper"
        )
        return result
    except Exception as e:
        st.error(f"Error upscaling image: {e}")
        return None

def new_model_inference(prompt, seed, randomize_seed, wallpaper_size, num_inference_steps, style_name):
    client = get_new_model_client()
    try:
        result = client.predict(
            prompt=prompt,
            seed=seed,
            randomize_seed=randomize_seed,
            wallpaper_size=wallpaper_size,
            num_inference_steps=num_inference_steps,
            style_name=style_name,
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

def run_image_api(image_path, param_0=0, param_1=0, param_3=True):
    client = get_upscale_client()
    try:
        result = client.predict(
            param_0=param_0,
            param_1=param_1,
            param_2=handle_file(image_path),
            param_3=param_3,
            api_name="/gpu_wrapped_execute_image"
        )
        return result
    except Exception as e:
        st.error(f"Error calling the Image API: {e}")
        return None

def run_video_api(image_path, video_path, param_2=True, param_3=True, param_4=True):
    client = get_upscale_client()
    try:
        result = client.predict(
            param_0=handle_file(image_path),
            param_1={"video": handle_file(video_path)},
            param_2=param_2,
            param_3=param_3,
            param_4=param_4,
            api_name="/gpu_wrapped_execute_video"
        )
        return result
    except Exception as e:
        st.error(f"Error calling the Video API: {e}")
        return None

def run_square_video_api(video_url):
    client = get_upscale_client()
    try:
        result = client.predict(
            video_path={"video": handle_file(video_url)},
            api_name="/is_square_video"
        )
        return result
    except Exception as e:
        st.error(f"Error calling the Square Video Check API: {e}")
        return None

# Ensure session state persists between reruns
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
    ["Generate Image from Prompt", "Face Swap", "Int. Dalle-3/F", "Gradio API Integration", "Image Upscaler", "Flux Ultimate"]
)

if app_mode == "Generate Image from Prompt":
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

            api_key = os.getenv("SEGMIND_API_KEY")  # Use environment variable for API key
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

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                image_bytes = io.BytesIO(response.content)
                st.session_state.face_swap_result = Image.open(image_bytes)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
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

elif app_mode == "Gradio API Integration":
    st.title("Gradio API Integration")

    image_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    if image_file:
        st.session_state.image_path = image_file

    if st.button("Process Image"):
        if st.session_state.image_path:
            with st.spinner('Processing image...'):
                result = run_image_api(st.session_state.image_path)
                if result:
                    st.image(result, caption="Processed Image", use_column_width=True)
        else:
            st.error("Please upload an image.")

elif app_mode == "Image Upscaler":
    st.title("Image Upscaler")

    uploaded_image = st.file_uploader("Upload an image to upscale", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.session_state.image_path = uploaded_image

    upscale_to = st.slider("Upscale to:", min_value=1, max_value=4, value=2)
    steps = st.slider("Steps:", min_value=1, max_value=100, value=20)
    noise_level = st.slider("Noise Level:", min_value=0.0, max_value=1.0, value=0.1)
    fidelity = st.slider("Fidelity:", min_value=0.0, max_value=1.0, value=0.5)
    seed = st.number_input("Seed:", min_value=0, max_value=10000, value=42)
    guidance_scale = st.slider("Guidance Scale:", min_value=0.0, max_value=20.0, value=7.5)
    sampler = st.selectbox("Sampler:", options=["euler", "heun", "dpm2"])

    if st.button("Upscale Image"):
        if st.session_state.image_path:
            with st.spinner('Upscaling image...'):
                upscaled_image = upscale_image(
                    st.session_state.image_path,
                    upscale_to,
                    steps,
                    noise_level,
                    fidelity,
                    seed,
                    guidance_scale,
                    sampler
                )
                if upscaled_image:
                    st.session_state.upscaled_image = upscaled_image

    if st.session_state.upscaled_image:
        st.image(st.session_state.upscaled_image, caption="Upscaled Image", use_column_width=True)
        buffer = io.BytesIO()
        st.session_state.upscaled_image.save(buffer, format="JPEG")
        buffer.seek(0)

        st.download_button(
            label="Download Upscaled Image",
            data=buffer,
            file_name="upscaled_image.jpg",
            mime="image/jpeg"
        )

elif app_mode == "Flux Ultimate":
    st.title("Flux Ultimate")

    prompt = st.text_input("Enter a prompt:")
    seed = st.number_input("Seed:", min_value=0, max_value=10000, value=42)
    randomize_seed = st.checkbox("Randomize Seed", value=True)
    wallpaper_size = st.selectbox("Wallpaper Size:", ["1024x768", "1280x720", "1920x1080"])
    num_inference_steps = st.slider("Number of Inference Steps:", min_value=1, max_value=100, value=20)
    style_name = st.text_input("Style Name:")

    if st.button("Generate with Flux Ultimate"):
        if prompt:
            with st.spinner('Generating with Flux Ultimate...'):
                new_model_result = new_model_inference(
                    prompt,
                    seed,
                    randomize_seed,
                    wallpaper_size,
                    num_inference_steps,
                    style_name
                )
                if new_model_result:
                    st.session_state.new_model_result = new_model_result

    if st.session_state.new_model_result:
        st.image(st.session_state.new_model_result, caption="Generated Image", use_column_width=True)
        buffer = io.BytesIO()
        st.session_state.new_model_result.save(buffer, format="JPEG")
        buffer.seek(0)

        st.download_button(
            label="Download Generated Image",
            data=buffer,
            file_name="flux_generated_image.jpg",
            mime="image/jpeg"
        )

# Additional code for video and square video API functionalities can be similarly updated and included.
