import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import os
from dotenv import load_dotenv

# تحميل مفتاح Gemini من .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# تحميل الموديل YOLO
model = YOLO("best.pt")   # اسم ملف الويتس اللي رفعته

# ===================== Sidebar =====================
st.sidebar.title("👥 About the Team")

team_members = {
    "Hossam": "images/hossam.jpg",
    "Martin ": "images/martin.jpg",
    "Mirna ": "images/mirna.jpg",
    "Mostafa": "images/mostafa.jpg",
    "Salma": "images/salma.jpg",
}

for name, img_path in team_members.items():
    try:
        st.sidebar.image(img_path, width=80, caption=name)
    except:
        st.sidebar.write(f"👤 {name}")

st.sidebar.markdown("---")
st.sidebar.title("📌 About the Project")
st.sidebar.write("""
- **Model:** YOLOv8 (best.pt)  
- **Data:** Google Landmarks Dataset  
- **API:** Google Gemini API for cultural info  
- **AR:** Streamlit frontend with AR-style info overlay  
""")

# ===================== Main App =====================
st.title("🏙️ AI City Explorer")

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة الأصلية أولًا
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # نسخة من الصورة للرسم عليها
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # توقع الكلاسات
    results = model.predict(image)

    detected_classes = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0]) * 100

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # رسم البوكس
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # رسم اسم الكلاس والـ accuracy
            text = f"{cls_name} {conf:.1f}%"
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
            draw.text((x1, y1 - text_height), text, fill="white", font=font)

            detected_classes.append((cls_name, conf))

    # عرض الصورة بعد الـ detection
    st.image(image_with_boxes, caption="Detected Image with Boxes", use_column_width=True)

    # عرض معلومات Gemini
    if detected_classes:
        st.subheader("✅ Detected Classes with Gemini Info")
        for cls_name, conf in detected_classes:
            with st.expander(f"📌 {cls_name} (Accuracy: {conf:.2f}%)", expanded=False):
                model_gemini = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"Write a short cultural description about {cls_name} in Egypt for a tourist app."
                response = model_gemini.generate_content(prompt)
                st.markdown(f"**📝 Gemini Info about {cls_name}:**")
                st.write(response.text)
    else:
        st.warning("⚠️ No classes detected.")
