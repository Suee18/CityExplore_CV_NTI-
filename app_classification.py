import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import google.generativeai as genai
import os
from dotenv import load_dotenv

# تحميل مفتاح Gemini من .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# تحميل موديل ResNet18 مدرب مسبقًا على مشروعك
model = models.resnet18()
num_classes = 35  # عدل حسب عدد الكلاسات عندك
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("egyptian_landmarks_classification_model.pth", map_location="cpu"))
model.eval()

# تحويل الصورة لتنسور ResNet18
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# أسماء الكلاسات
class_names = [
    "Colossoi of Memnon",
    "Sphinx",
    "Great Pyramids of Giza",
    "Mask of Tutankhamun",
    "Bent Pyramid of King Sneferu",
    "Pyramid of Djoser",
    "Nefertiti",
    "Amenhotep III and Tiye",
    "bust of Ramesses II",
    "Statue of King Zoser",
    "King Thutmose III",
    "Isis with her child",
    "Akhenaten",
    "Statue of Tutankhamun",
    "Statue of Ankhesenamun",
    "Statue of King Ramses II Luxor Temple",
    "Standing Statue of King Ramses II",
    "Hatshepsut face",
    "Statue of King Ramses II Grand Egyptian Museum",
    "Amenhotep III",
    "Head Statue of Amenhotep iii",
    "Colossal Statue of Queen Hatshepsut",
    "Statue of Khafre",
    "Colossal Statue of King Senwosret IlI",
    "Statue of King Sety Il Holding Standards",
    "Statue of Amenmhat I",
    "Granite Statue of Tutankhamun",
    "Seated Statue of Amenhotep III",
    "Colossal Statue of Middle Kingdom King",
    "Obelsik Tip of Hatshepsut",
    "Colossal Statue of Hormoheb",
    "Sphinx of Kings Ramesses ll - Merenptah",
    "Statue of God Ptah Ramesses ll Goddess Sekhmet",
    "Statue of Snefru",
    "Menkaure Statue"
] 

st.title("🏙️ AI City Explorer - Classification")

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # تجهيز الصورة
    input_tensor = preprocess(image).unsqueeze(0)  # batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, cls_idx = torch.max(probs, 0)

    cls_name = class_names[cls_idx]
    conf = conf.item() * 100

    # عرض النتيجة
    st.subheader("✅ Detected Class")
    with st.expander(f"📌 {cls_name} (Accuracy: {conf:.2f}%)", expanded=True):
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Write a short cultural description about {cls_name} in Egypt for a tourist app."
        response = model_gemini.generate_content(prompt)
        st.markdown(f"**📝 Gemini Info about {cls_name}:**")
        st.write(response.text)
