import streamlit as st 
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import re


# تحميل مفتاح Gemini من .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load YOLO model
yolo_model = YOLO("best.pt")   # غيّر المسار حسب الموديل عندك

# Load Torch classification model
model_save_path = "egyptian_landmarks_classification_model.pth"
torch_model = models.resnet18(pretrained=False)
num_ftrs = torch_model.fc.in_features
torch_model.fc = nn.Linear(num_ftrs, 35)  # 35 classes
torch_model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
torch_model.eval()

# Class labels
groundTruth = [
    'Colossoi of Memnon','Sphinx','King Thutmose III','Isis with her child','Akhenaten',
    'Statue of Tutankhamun','Statue of Ankhesenamun','Statue of King Ramses II Luxor Temple',
    'Standing Statue of King Ramses II','Hatshepsut face','Statue of King Ramses II Grand Egyptian Museum',
    'Amenhotep III','Great Pyramids of Giza','Head Statue of Amenhotep iii',
    'Colossal Statue of Queen Hatshepsut','Statue of Khafre','Colossal Statue of King Senwosret IlI',
    'Statue of King Sety Il Holding Standards','Statue of Amenmhat I','Granite Statue of Tutankhamun',
    'Seated Statue of Amenhotep III','Colossal Statue of Middle Kingdom King','Obelsik Tip of Hatshepsut',
    'Mask of Tutankhamun','Colossal Statue of Hormoheb','Sphinx of Kings Ramesses ll - Merenptah',
    'Statue of God Ptah Ramesses ll Goddess Sekhmet','Statue of Snefru','Menkaure Statue',
    'Bent Pyramid of King Sneferu','Pyramid of Djoser','Nefertiti','Amenhotep III and Tiye',
    'bust of Ramesses II','Statue of King Zoser'
]

# Preprocessing transforms
image_size = 224
val_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===================== Main App =====================
st.title("🏙️ City Explorer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Step 1: Show original uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Step 2: Run YOLO and draw boxes
    results = yolo_model.predict(img_np)
    img_draw = img_np.copy()
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Bright neon green with thicker border
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (57, 255, 20), 12)

    # 👉 Side by side display
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Uploaded Image", use_container_width=True)
    with col2:
        st.image(img_draw, caption="Image with YOLO Boxes", use_container_width=True)

    # Step 3: Crop each box, classify with Torch, show results + Gemini
    st.subheader("Predictions with Gemini Info:")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img_np[y1:y2, x1:x2]

        # Apply transforms
        crop_pil = Image.fromarray(crop)
        crop_tensor = val_transforms(crop_pil).unsqueeze(0)

        # Torch model prediction
        with torch.no_grad():
            output = torch_model(crop_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probs, 0)
            label = groundTruth[pred]


        conf_percent = conf.item() * 100
        st.write(f"🔮 Prediction: *{label}* (Accuracy: {conf_percent:.2f}%)")

        # Gemini structured JSON description
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        You are an AI that provides tourist information about Egyptian landmarks. 
        Return ONLY valid JSON in this format:

        {{
          "paragraph": "A 5-sentence cultural description about {label}.",
          "location": "City, Egypt",
          "years_old": "Approximate age in years",
          "Era": "Historical era ",
          "category": "Temple, Pyramid, Statue, or Museum Artifact",
          "fun_fact": "One fun/interesting fact about it.",
          "Average visitors per year": "Number of visitors",
          "Visitor_Info": "Visiting hours, entry fee, best time to visit",
          "Dimensions": "Height, Width, Depth in meters",
          "Material": "Primary material used (e.g., limestone, granite, etc.)"

        }}
        """
        response = model_gemini.generate_content(prompt)   ### FIX (was model not defined)
        raw_response = response.text.strip()

       # Parse the JSON safely

        raw_text = response.text

        # Remove code block markers if they exist
        clean_text = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL)

        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError:
            st.error("❌ Model response is not valid JSON")
            st.write(raw_text)  # fallback raw output
        else:
           # Title
            st.markdown(f"<h1 style='text-align: center; color: beige; font-weight: bold;'>{label}</h1>",
                    unsafe_allow_html=True)

            # Description paragraph
            st.markdown(f"**🌍 Cultural Description:** {data.get('paragraph', '')}")

            # Grid layout
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**📍 Location:** {data.get('location', 'N/A')}")
                st.markdown(f"**📅 Age:** {data.get('years_old', 'N/A')}")
                st.markdown(f"**📜 Era:** {data.get('Era', 'N/A')}")
                st.markdown(f"**🏷️ Category:** {data.get('category', 'N/A')}")
                st.markdown(f"**📐 Dimensions:** {data.get('Dimensions', 'N/A')}")
                st.markdown(f"**🪨 Material:** {data.get('Material', 'N/A')}")
            with col2:
                st.markdown(f"**✨ Fun Fact:** {data.get('fun_fact', 'N/A')}")
                st.markdown(f"**👥 Visitors/year:** {data.get('Average visitors per year', 'N/A')}")
                st.markdown(f"**🕓 Visiting Info:** {data.get('Visitor_Info', 'N/A')}")
              

            st.divider()





# Sidebar Section
# ========================
# Sidebar Section
# ========================
with st.sidebar:
    st.header("⚙️ About the Model")
    st.info("This is a **classifer model** pre-trained to classify Egyptian monuments")
    st.markdown(
        """
        ### 🔧 Key Details: Two-Stage AI Pipeline
        
        Our system employs a sophisticated **dual-model architecture** for accurate Egyptian landmark identification:
        
        **Stage 1 - YOLO Object Detection:**
        - Analyzes uploaded images to locate and isolate landmarks using bounding boxes
        - Handles complex scenes with multiple objects or cluttered backgrounds
        - Crops detected regions for precise classification
        
        **Stage 2 - ResNet18 Classification:**
        - Processes cropped landmark images through a fine-tuned neural network
        - Trained on 35 distinct Egyptian monument classes
        - Achieves high accuracy by focusing on isolated landmark features
        
        **Integration Benefits:**
        - **Robustness**: Works with any image composition or background
        - **Precision**: Dual-stage approach ensures accurate identification
        - **Scalability**: Can detect multiple landmarks in a single image
        - **Intelligence**: Combines computer vision with cultural knowledge via Gemini AI
        
        This pipeline transforms raw tourist photos into rich, educational experiences about Egypt's cultural heritage.
        """,
        unsafe_allow_html=True
    )

    st.header("🖼️ Dataset Info")
    st.write("RoboFlow dataset for the pretrained model")
    st.info("""
        - 35 classes
        - 10646 images
    """)

    # ========================
    # Developers Section
    # ========================
    st.header("👩‍💻 Developers")

    developers = [
        {"name": "Martin", "img": "images/martin.jpg", "linkedin": "https://www.linkedin.com/in/martin-emad-39875429b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app"},
        {"name": "Mirna", "img": "images/mirna.jpg", "linkedin": "https://www.linkedin.com/in/mirna-nageh-botros?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"},
        {"name": "Mostafa", "img": "images/Mostafa.jpg", "linkedin": "https://www.linkedin.com/in/mustafa-mohamed131/"},
        {"name": "Hossam", "img": "images/hossam.jpg", "linkedin": "https://linkedin.com/in/hossam"},
        {"name": "Salma", "img": "images/salma.jpg", "linkedin": "https://linkedin.com/in/salma"},
    ]

    for dev in developers:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                st.image(dev['img'], width=80)
            except:
                st.write("📷")  # Fallback if image not found
        
        with col2:
            st.markdown(f"**{dev['name']}**")
            st.markdown(f"[🔗 LinkedIn]({dev['linkedin']})")
        
        st.markdown("---")

