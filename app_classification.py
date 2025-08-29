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

# Streamlit UI
st.title("YOLO + Torch + Gemini Explorer 🚀")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Step 1: Show original uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    st.image(img, caption="Original Uploaded Image", use_container_width=True)

    # Step 2: Run YOLO and draw boxes
    results = yolo_model.predict(img_np)
    img_draw = img_np.copy()
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
          "category": "Temple, Pyramid, Statue, or Museum Artifact",
          "fun_fact": "One fun/interesting fact about it."
        }}
        """
        response = model_gemini.generate_content(prompt)   ### FIX (was model not defined)
        raw_response = response.text.strip()

        try:
            info = json.loads(raw_response)   ### FIX (was data/info mismatch)
        except json.JSONDecodeError:
            print("⚠️ Gemini did not return valid JSON:", raw_response)
            info = {
                "paragraph": raw_response,
                "location": "N/A",
                "years_old": "N/A",
                "category": "N/A",
                "fun_fact": "N/A"
            }

        # Show nicely in Streamlit
        st.markdown(f"### 🏛️ {label}")
        st.write(info["paragraph"])
        st.write(f"**📍 Location:** {info['location']}")
        st.write(f"**⏳ Age:** {info['years_old']} years old")
        st.write(f"**🏷️ Category:** {info['category']}")
        st.write(f"**✨ Fun Fact:** {info['fun_fact']}")
