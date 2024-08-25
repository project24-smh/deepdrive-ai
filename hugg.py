import os
import tempfile
import time
import streamlit as st
from gradio_client import Client, handle_file
import requests
from PIL import Image
import io
import base64

# Initialize the Gradio client
client = Client("KwaiVGI/LivePortrait")

# Hugging Face API URL and headers
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def generate_image(prompt):
    try:
        image_bytes = query({"inputs": prompt})
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def image_to_base64(img):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to run the Image API
def run_image_api(image_path, param_0=0, param_1=0, param_3=True):
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

# Function to run the Video API
def run_video_api(image_path, video_path, param_2=True, param_3=True, param_4=True):
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

# Function to run the Square Video Check API
def run_square_video_api(video_url):
    try:
        result = client.predict(
            video_path={"video": handle_file(video_url)},
            api_name="/is_square_video"
        )
        return result
    except Exception as e:
        st.error(f"Error calling the Square Video Check API: {e}")
        return None

# Ensure that session state persists between reruns
if "image_path" not in st.session_state:
    st.session_state.image_path = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

# Main app layout
st.sidebar.title("Select an Option")
app_mode = st.sidebar.radio(
    "Choose an option",
    ["Generate Image from Prompt", "Face Swap", "Int. Dalle-3/F", "Gradio API Integration"]
)

if app_mode == "Generate Image from Prompt":
    st.title("Generate Image from Prompt")

    # Prompt input
    prompt = st.text_input("Enter a prompt for the image:")

    # Add a button to generate the image
    if st.button("Generate Image"):
        if prompt:
            with st.spinner('Generating image...'):
                img = generate_image(prompt)
                if img:
                    st.image(img, caption="Generated Image", use_column_width=True)
                    
                    # Prepare image for download
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    buffer.seek(0)

                    st.download_button(
                        label="Download Generated Image",
                        data=buffer,
                        file_name="generated_image.jpg",
                        mime="image/jpeg"
                    )
        else:
            st.error("Please enter a prompt.")

elif app_mode == "Face Swap":
    st.title("Face Swap App")

    # Upload source image
    st.subheader("Upload Source Image")
    source_image = st.file_uploader("Choose a source image...", type=["jpg", "jpeg"])

    # Upload target image
    st.subheader("Upload Target Image")
    target_image = st.file_uploader("Choose a target image...", type=["jpg", "jpeg"])

    if source_image and target_image:
        st.image(source_image, caption="Source Image", use_column_width=True)
        st.image(target_image, caption="Target Image", use_column_width=True)

        if st.button("Swap Faces"):
            source_img = Image.open(source_image)
            target_img = Image.open(target_image)

            # Convert images to base64 with high quality
            source_b64 = image_to_base64(source_img)
            target_b64 = image_to_base64(target_img)

            # API call for face swapping
            api_key = "SG_498e9675cc2805a5"  # Directly included API key
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
                # Handle response as an image
                image_bytes = io.BytesIO(response.content)
                swapped_img = Image.open(image_bytes)
                st.image(swapped_img, caption="Face Swapped Image", use_column_width=True)

                # Prepare image for download
                buffer = io.BytesIO()
                swapped_img.save(buffer, format="JPEG")
                buffer.seek(0)

                st.download_button(
                    label="Download Face Swapped Image",
                    data=buffer,
                    file_name="face_swapped_image.jpg",
                    mime="image/jpeg"
                )
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

elif app_mode == "Int. Dalle-3/F":
    st.title("Dalle-3 Integration")

    # Embed the external page in the main page
    st.markdown(
        f"""
        <iframe src="https://nymbo-flux-1-dev-serverless.hf.space/" 
        width="100%" height="800" frameborder="0" scrolling="auto"></iframe>
        """,
        unsafe_allow_html=True
    )

elif app_mode == "Gradio API Integration":
    st.title("Gradio Client API Integration")

    # API Selection
    api_options = ["Execute Image", "Execute Video", "Check Square Video"]
    api_choice = st.selectbox("Choose an API to call", api_options, index=1)  # Default to "Execute Video"

    # Image Upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        temp_dir = tempfile.mkdtemp()
        st.session_state.image_path = os.path.join(temp_dir, uploaded_image.name)
        with open(st.session_state.image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        st.image(st.session_state.image_path, caption="Uploaded Image", use_column_width=True)

    # Video Upload or URL Input
    video_input_type = st.radio("Select Video Input Type", ["Upload", "URL"])

    if video_input_type == "Upload":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
        if uploaded_video is not None:
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

    # Wait time counter
    wait_time = st.empty()
    if st.button("Run API"):
        wait_time.text("Processing... Please wait.")
        start_time = time.time()

        if api_choice == "Execute Image":
            if st.session_state.image_path is not None:
                result = run_image_api(st.session_state.image_path)
                if result:
                    st.image(result[0], caption="Result Image 1")
                    st.image(result[1], caption="Result Image 2")
            else:
                st.error("Please upload an image.")

        elif api_choice == "Execute Video":
            if st.session_state.image_path is not None and st.session_state.video_path is not None:
                result = run_video_api(st.session_state.image_path, st.session_state.video_path)
                if result:
                    st.video(result[0]["video"], format="video/mp4")
                    st.video(result[1]["video"], format="video/mp4")
            else:
                st.error("Please upload both an image and a video or provide a video URL.")

        elif api_choice == "Check Square Video":
            if st.session_state.video_path is not None:
                result = run_square_video_api(st.session_state.video_path)
                st.write("Is Square Video:", result)
            else:
                st.error("Please provide a video URL.")

        end_time = time.time()
        wait_time.text(f"Processing completed in {end_time - start_time:.2f} seconds.")

