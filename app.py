import streamlit as st
from PIL import Image
import google.generativeai as genai
from ultralytics import YOLO
import torch

# --- Page Configuration ---
st.set_page_config(
    page_title="🏛️ Landmark Lens",
    page_icon="🌍",
    layout="wide"
)

# --- Gemini API Configuration ---
# Configure the Gemini API with the key from Streamlit's secrets
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except (KeyError, AttributeError):
    st.error("🔑 Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- Model Loading ---

# Function to load your fine-tuned YOLOv8 model
# @st.cache_resource is used to load the model only once, improving performance
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8 model from the 'best.pt' file."""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.info("Please make sure the 'best.pt' file is in the same directory as 'app.py'.")
        return None

# Function to get a description from the Gemini Pro model (text-only)
def get_gemini_description(landmark_name):
    """
    Takes a landmark name and returns an interesting description from Gemini.
    """
    # We use 'gemini-pro' here since we are only sending text
    model = genai.GenerativeModel('gemini-1.5-flash-latest')    # A more engaging prompt for a tour guide
    prompt = f"""
    You are an expert tour guide with a passion for history and architecture.
    A tourist has just identified the following location: "{landmark_name}".

    Please provide a captivating and informative description (about 3-4 sentences) that includes:
    1. A key historical fact.
    2. Its significance or what makes it famous.
    3. A fun or interesting tidbit.

    Make your response engaging and easy to read.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not retrieve information from Gemini: {e}"

# Load the YOLO model
yolo_model = load_yolo_model()

# --- Streamlit App UI ---

st.title("🏛️ Landmark Lens: Your AI Tour Guide")
st.write("Upload an image of a landmark, and my custom-trained vision model will identify it, then our AI guide will tell you all about it!")

uploaded_file = st.file_uploader("Upload your landmark image", type=["jpg", "jpeg", "png"])

if yolo_model and uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Create two columns for a clean layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Your Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Analysis Results")
        
        # When the button is clicked, perform detection and description
        if st.button("Identify and Describe Landmark"):
            with st.spinner("🔍 Analyzing with YOLOv8..."):
                # Perform object detection on the image
                results = yolo_model(image)
                
                # Check if any objects were detected
                if results and results[0].boxes:
                    # Get the name of the top detected class
                    top_detection = results[0].boxes[0]
                    class_id = int(top_detection.cls)
                    landmark_name = yolo_model.names[class_id]
                    confidence = float(top_detection.conf)

                    st.success(f"**Identified Landmark:** {landmark_name}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    with st.spinner("✍️ Generating description with Gemini..."):
                        # Get the description from Gemini
                        description = get_gemini_description(landmark_name)
                        st.markdown("---")
                        st.subheader("📜 AI Tour Guide Says:")
                        st.markdown(description)
                else:
                    st.error("Sorry, my model couldn't identify a known landmark in this image. Please try another one!")

elif yolo_model is None:
    st.warning("YOLO Model could not be loaded. The app is not functional.")
else:
    st.info("Please upload an image to begin.")