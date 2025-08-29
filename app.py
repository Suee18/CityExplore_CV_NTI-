import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# تحميل مفتاح Gemini من .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# تحميل الموديل YOLO
model = YOLO("best.pt")   # اسم ملف الويتس اللي رفعته

st.title("🏙️ AI City Explorer")

# رفع صورة
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # فتح الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # توقع الكلاسات
    results = model.predict(image)

    # استخراج أول كلاس فقط
    first_class = None
    for r in results:
        if len(r.boxes) > 0:
            box = r.boxes[0]  # أول بوكس
            cls_id = int(box.cls[0])        
            cls_name = r.names[cls_id]      
            conf = float(box.conf[0]) * 100 
            first_class = (cls_name, conf)
            break  # نوقف عند أول كلاس

    # عرض النتيجة
    if first_class:
        cls_name, conf = first_class
        st.subheader("✅ Detected Class")
        with st.expander(f"📌 {cls_name} (Accuracy: {conf:.2f}%)", expanded=True):
            # استعلام واحد فقط للـ Gemini API
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Write a short cultural description about {cls_name} in Egypt for a tourist app."
            response = model_gemini.generate_content(prompt)

            st.markdown(f"**📝 Gemini Info about {cls_name}:**")
            st.write(response.text)
    else:
        st.warning("⚠️ No classes detected.")
