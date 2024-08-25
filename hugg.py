import os
import io
import time
import tempfile
import requests
import streamlit as st
from PIL import Image
from gradio_client import Client, handle_file

# Initialize the Gradio client
client = Client("KwaiVGI/LivePortrait")

# Hugging Face API URLs and Authorization headers
API_URL_FLUX_SCHNELL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
API_URL_FLUX_DEV = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
HF_HEADERS = {"Authorization": "Bearer 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"}

def query_huggingface_api(prompt, api_url):
    """Query the Hugging Face API to generate an image from the provided prompt."""
    response = requests.post(api_url, headers=HF_HEADERS, json={"inputs": prompt})
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

def run_image_api(image_path, param_0=0, param_1=0, param_3=True):
    """Call the Gradio Image API."""
    return client.predict(
        param_0=param_0,
        param_1=param_1,
        param_2=handle_file(image_path),
        param_3=param_3,
        api_name="/gpu_wrapped_execute_image"
    )

def run_video_api(image_path, video_path, param_2=True, param_3=True, param_4=True):
    """Call the Gradio Video API."""
    return client.predict(
        param_0=handle_file(image_path),
        param_1={"video": handle_file(video_path)},
        param_2=param_2,
        param_3=param_3,
        param_4=param_4,
        api_name="/gpu_wrapped_execute_video"
    )

def check_square_video(video_url):
    """Check if a video is square using the Gradio API."""
    return client.predict(
        video_path={"video": handle_file(video_url)},
        api_name="/is_square_video"
    )

def convert_image_to_base64(img):
    """Convert a PIL image to a base64 encoded string."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Ensure session state persists across reruns
if "image_path" not in st.session_state:
    st.session_state.image_path = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

# Sidebar options
st.sidebar.title("Options")
app_mode = st.sidebar.radio(
    "Choose an option",
    ["Generate Image", "Face Swap", "Dalle-3 Integration", "Gradio API Integration"]
)

if app_mode == "Generate Image":
    st.title("Generate Image from Prompt")

    prompt = st.text_input("Enter a prompt:")
    api_choice = st.selectbox("Select API", ["FLUX.1-schnell", "FLUX.1-dev"], index=1)
    api_url = API_URL_FLUX_DEV if api_choice == "FLUX.1-dev" else API_URL_FLUX_SCHNELL

    if st.button("Generate Image"):
        if prompt:
            with st.spinner('Generating image...'):
                try:
                    img = query_huggingface_api(prompt, api_url)
                    st.image(img, caption="Generated Image", use_column_width=True)

                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    buffer.seek(0)
                    st.download_button("Download Image", data=buffer, file_name="generated_image.jpg", mime="image/jpeg")

                except Exception as e:
                    st.error(f"Failed to generate image: {e}")
        else:
            st.error("Please enter a prompt.")

elif app_mode == "Face Swap":
    st.title("Face Swap")

    source_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg"])
    target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg"])

    if source_image and target_image:
        st.image(source_image, caption="Source Image", use_column_width=True)
        st.image(target_image, caption="Target Image", use_column_width=True)

        if st.button("Swap Faces"):
            try:
                source_img = Image.open(source_image)
                target_img = Image.open(target_image)
                source_b64 = convert_image_to_base64(source_img)
                target_b64 = convert_image_to_base64(target_img)

                data = {
                    "source_img": source_b64,
                    "target_img": target_b64,
                    "input_faces_index": 0,
                    "source_faces_index": 0,
                    "face_restore": "codeformer-v0.1.0.pth",
                    "base64": False
                }

                headers = {'x-api-key': "SG_498e9675cc2805a5", 'Content-Type': 'application/json'}
                response = requests.post("https://api.segmind.com/v1/faceswap-v2", json=data, headers=headers)

                if response.status_code == 200:
                    swapped_img = Image.open(io.BytesIO(response.content))
                    st.image(swapped_img, caption="Face Swapped Image", use_column_width=True)

                    buffer = io.BytesIO()
                    swapped_img.save(buffer, format="JPEG")
                    buffer.seek(0)
                    st.download_button("Download Image", data=buffer, file_name="face_swapped_image.jpg", mime="image/jpeg")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Failed to swap faces: {e}")

elif app_mode == "Dalle-3 Integration":
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

    api_choice = st.selectbox("Select API", ["Execute Image", "Execute Video", "Check Square Video"], index=1)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        temp_dir = tempfile.mkdtemp()
        st.session_state.image_path = os.path.join(temp_dir, uploaded_image.name)
        with open(st.session_state.image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        st.image(st.session_state.image_path, caption="Uploaded Image", use_column_width=True)

    video_input_type = st.radio("Select Video Input Type", ["Upload", "URL"])

    if video_input_type == "Upload":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
        if uploaded_video:
            temp_dir = tempfile.mkdtemp()
            st.session_state.video_path = os.path.join(temp_dir, uploaded_video.name)
            with open(st.session_state.video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            st.video(st.session_state.video_path, format="video/mp4")
            st.write("Uploaded Video")
    else:
        video_url = st.text_input("Enter Video URL")
        if video_url:
            st.session_state.video_path = video_url

    if st.button("Run API"):
        wait_time = st.empty()
        wait_time.text("Processing... Please wait.")
        start_time = time.time()

        if api_choice == "Execute Image":
            if st.session_state.image_path:
                result = run_image_api(st.session_state.image_path)
                if result:
                    st.image(result[0], caption="Result Image 1")
                    st.image(result[1], caption="Result Image 2")
                else:
                    st.error("Failed to process the image.")

        elif api_choice == "Execute Video":
            if st.session_state.image_path and st.session_state.video_path:
                result = run_video_api(st.session_state.image_path, st.session_state.video_path)
                if result:
                    st.video(result[0], format="video/mp4")
                else:
                    st.error("Failed to process the video.")

        elif api_choice == "Check Square Video":
            if st.session_state.video_path:
                result = check_square_video(st.session_state.video_path)
                if result:
                    st.write(f"Is Square Video: {result}")
                else:
                    st.error("Failed to check the video.")
