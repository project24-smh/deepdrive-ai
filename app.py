import streamlit as st
import replicate
import time
import os
import requests
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from a .env file (if needed)
load_dotenv()

st.title("Imagine Your Dream")
st.text("Made by Samiul")

# Add a text input for the API token
api_token = st.text_input("Enter your API token", type="password")

# Prompt input
prompt = st.text_input("Enter a prompt for the image:")

# Add a button to generate the image
if st.button("Generate Image"):
    if api_token:
        os.environ["REPLICATE_API_TOKEN"] = api_token
        with st.spinner('Generating image...'):
            start_time = time.time()
            output = replicate.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": prompt,
                    "num_outputs": 1,
                    "output_format": "jpg",
                    "disable_safety_checker": True  # Always enable the safety checker
                }
            )

            # Assuming the output is a URL to the image
            image_url = output[0]  # Get the first (and only) image URL
            response = requests.get(image_url)
            image = BytesIO(response.content)

            # Display the generated image
            st.image(image)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Image generated in {elapsed_time:.2f} seconds")

            # Add a download button for the image
            st.download_button(
                label="Download Image",
                data=image,
                file_name="generated_image.jpg",
                mime="image/jpeg"
            )
    else:
        st.error("Please enter your API token.")
