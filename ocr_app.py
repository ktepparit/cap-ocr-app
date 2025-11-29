import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import re

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng OCR Scanners", page_icon="⚡", layout="centered")

# --- โหลดโมเดล (Cache) ---
@st.cache_resource
def load_model():
    return easyocr.Reader(['en'], gpu=False)

with st.spinner('กำลังเตรียมระบบอ่านรหัส...'):
    reader = load_model()

# --- ฟังก์ชันช่วย: หมุนภาพ (ใช้ PIL แทน OpenCV) ---
def rotate_image(image, angle):
    if angle == 0: return image
    # expand=True เพื่อให้รูปไม่โดนตัดขอบตอนหมุน
    return image.rotate(-angle, expand=True) 

# --- ฟังก์ชันหลัก: อ่านและคัดกรอง ---
def smart_read(image_pil):
    # 1. ปรับภาพให้ชัดขึ้น (Preprocessing แบบไม่ใช้ OpenCV)
    # แปลงเป็นขาวดำ
    img_processed = image_pil.convert('L') 
    # เพิ่มความคมชัด (Contrast)
    enhancer = ImageEnhance.Contrast(img_processed)
    img_processed = enhancer.enhance(2.0) # เพิ่ม Contrast 2 เท่า
    
    candidates = []

    # วนลูปหมุนภาพ 4 ทิศ (0, 90, 180, 270 องศา)
    for angle in [0, 90, 180, 270]:
        rotated_img = rotate_image(img_processed, angle)
        
        # แปลงเป็น numpy array เพื่อส่งให้ EasyOCR
        img_np = np.array(rotated_img)
        
        # อ่านค่า (detail=0 เอาแค่ text)
        results = reader.readtext(img_np, detail=0)
        
        # รวมผลลัพธ์
        full_text = "".join(results)
        
        # --- กรองเข้มงวด (Regex) ---
        full_text = full_text.upper()
        # เก็บเฉพาะ A-Z และ 0-9 
        clean_text = re.sub(r'[^A-Z0-9]', '', full_text)
        
        # หา Pattern 9 ตัวเป๊ะๆ
        matches = re.findall(r'[A-Z0-9]{9}', clean_text)
        
        for match in matches:
            return match # เจอแล้วส่งกลับเลย

        # ถ้าไม่เจอ 9 ตัวเป๊ะ ให้เก็บค่าที่ใกล้เคียงไว้ (7-12 ตัวอักษร)
        if len(clean_text) >= 7 and len(clean_text) <= 12:
             candidates.append(clean_text)

    # ถ้าวนครบแล้วไม่เจอ ให้เอาตัวที่ยาวที่สุดที่หาได้
    if candidates:
        return max(candidates, key=len)
    
    return None

# --- ส่วนแสดงผลโลโก้ ---
try:
    st.image("banner.png", use_column_width=True)
except FileNotFoundError:
    st.title("⚡ ระบบสแกนรหัสฝาขวด")

# --- ส่วนเนื้อหาหลัก ---
st.write("---")
st.info("ℹ️ โหมด Lite: ทำงานเร็วขึ้น รองรับภาพเอียงและกลับหัว")

tab1, tab2 = st.tabs(["📂 อัปโหลดหลายรูป", "📷 ถ่ายรูป"])

# ================= TAB 1: Batch Upload =================
with tab1:
    uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"กำลังประมวลผล {len(uploaded_files)} รูปภาพ...")
        st.markdown("---")
        
        for i, uploaded_file in enumerate(uploaded_files):
            col1, col2 = st.columns([1, 3])
            image = Image.open(uploaded_file)

            with col1:
                st.image(image, width=100, caption=f"รูปที่ {i+1}")

            with col2:
                with st.spinner('🔄 กำลังสแกน 4 ทิศทาง...'):
                    final_code = smart_read(image)
                    
                    if final_code:
                        st.code(final_code, language=None)
                        if len(final_code) != 9:
                            st.caption(f"⚠️ พบ {len(final_code)} ตัวอักษร (ควรเป็น 9)")
                        else:
                            st.caption("✅ รหัสสมบูรณ์")
                    else:
                        st.error("❌ อ่านค่าไม่ได้")
            st.markdown("---")

# ================= TAB 2: Camera =================
with tab2:
    camera_image = st.camera_input("ถ่ายรูปฝาขวด")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.write("ผลลัพธ์:")
        with st.spinner('🔄 กำลังสแกน 4 ทิศทาง...'):
            final_code = smart_read(image)
            if final_code:
                st.code(final_code, language=None)
                if len(final_code) == 9:
                    st.caption("✅ รหัสสมบูรณ์")
                else:
                     st.caption(f"⚠️ ค่าที่อ่านได้: {len(final_code)} ตัวอักษร")
            else:
                st.warning("ไม่พบรหัสที่ชัดเจน")
