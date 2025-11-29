import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import re

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng OCR Scanners", page_icon="⚡", layout="centered")

# --- โหลดโมเดล ---
@st.cache_resource
def load_model():
    return easyocr.Reader(['en'], gpu=False)

with st.spinner('กำลังเตรียมระบบ...'):
    reader = load_model()

# --- Logic: ตรวจสอบเงื่อนไข 9 หลัก + เลข 0/2 ตัว ---
def is_valid_pattern(text):
    if len(text) != 9:
        return False
    digit_count = sum(c.isdigit() for c in text)
    # ต้องมีตัวเลข 0 ตัว หรือ 2 ตัว เท่านั้น
    return digit_count == 0 or digit_count == 2

# --- Preprocessing (สูตรใหม่: Smart Resize) ---
def enhance_image_for_ocr(image):
    # 1. ตัดขอบ 18% (ลบ HDPE/ขอบฝา)
    width, height = image.size
    crop_val = 0.18
    image = image.crop((width*crop_val, height*crop_val, width*(1-crop_val), height*(1-crop_val)))
    
    # 2. [แก้จุดที่ทำแอพพัง] Smart Resize
    # บังคับความกว้างเป็น 1200px (ความละเอียดระดับ HD)
    # ใหญ่พอที่จะเห็นหยักตัว W แต่ไม่ใหญ่จน Server ระเบิด
    target_width = 1200
    if image.width != target_width:
        w_percent = (target_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((target_width, h_size), resample=Image.LANCZOS)
    
    # 3. แปลงเป็นขาวดำ
    image = image.convert('L')
    
    # 4. [แก้ตัว K อ่านเป็น I] Histogram Equalization
    # เกลี่ยแสงให้สม่ำเสมอ เพื่อกู้รายละเอียดในเงา
    image = ImageOps.equalize(image)
    
    # 5. เพิ่ม Contrast (แต่ไม่เยอะเกินไปจนเส้นขาด)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)
    
    return image

# --- ฟังก์ชันหลัก ---
def smart_read(image_pil):
    # เตรียมภาพ
    processed_img = enhance_image_for_ocr(image_pil)
    
    candidates = []

    # วนลูปหมุน 4 ทิศ (0, 90, 180, 270)
    for angle in [0, 90, 180, 270]:
        if angle != 0:
            rotated = processed_img.rotate(-angle, expand=True)
        else:
            rotated = processed_img
            
        img_np = np.array(rotated)
        
        # อ่านค่า (Allowlist)
        results = reader.readtext(img_np, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        full_text = "".join(results).upper()
        clean_text = re.sub(r'[^A-Z0-9]', '', full_text)
        
        # 1. หา Pattern 9 ตัวเป๊ะ
        for i in range(len(clean_text) - 8):
            chunk = clean_text[i : i+9]
            if is_valid_pattern(chunk):
                return chunk # เจอของดี ส่งกลับเลย

        # 2. เก็บตัวสำรอง (8-10 ตัว)
        if len(clean_text) >= 8 and len(clean_text) <= 10:
            candidates.append(clean_text)

    # เลือกตัวที่ดีที่สุดจาก Candidates
    if candidates:
        # กรองหาตัวที่ผ่านเงื่อนไขเลข 0/2 ตัว
        priority_candidates = [c for c in candidates if is_valid_pattern(c)]
        if priority_candidates:
            return max(priority_candidates, key=len)
            
        # ถ้าไม่มีจริงๆ เอาตัวที่ยาวใกล้ 9 สุด
        return sorted(candidates, key=lambda x: abs(len(x) - 9))[0]
    
    return None

# --- ส่วนแสดงผลโลโก้ (150x201) ---
try:
    logo = Image.open("banner.png")
    logo_resized = logo.resize((150, 201))
    col_logo, col_space = st.columns([1, 2])
    with col_logo:
        st.image(logo_resized)
except FileNotFoundError:
    pass

# --- UI ---
st.write("---")
st.info("ℹ️ Mode: Smart HD (แก้ปัญหาแอพพัง + อ่าน W/K แม่นยำ)")

tab1, tab2 = st.tabs(["📂 อัปโหลดหลายรูป", "📷 ถ่ายรูป"])

# TAB 1: Batch
with tab1:
    uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"กำลังประมวลผล {len(uploaded_files)} รูป...")
        st.markdown("---")
        for i, uploaded_file in enumerate(uploaded_files):
            col1, col2 = st.columns([1, 3])
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, width=100, caption=f"รูปที่ {i+1}")
            with col2:
                with st.spinner('Scanning...'):
                    final_code = smart_read(image)
                    if final_code:
                        st.code(final_code, language=None)
                        d_c = sum(c.isdigit() for c in final_code)
                        if len(final_code) == 9 and (d_c == 0 or d_c == 2):
                            st.caption("✅ ถูกต้อง")
                        else:
                            st.caption(f"⚠️ ตรวจสอบ: {final_code}")
                    else:
                        st.error("❌ อ่านไม่ออก")
            st.markdown("---")

# TAB 2: Camera
with tab2:
    camera_image = st.camera_input("ถ่ายรูป")
    if camera_image is not None:
        image = Image.open(camera_image)
        with st.spinner('Scanning...'):
            final_code = smart_read(image)
            if final_code:
                st.code(final_code, language=None)
                d_c = sum(c.isdigit() for c in final_code)
                if len(final_code) == 9 and (d_c == 0 or d_c == 2):
                    st.caption("✅ ถูกต้อง")
                else:
                    st.caption(f"⚠️ ตรวจสอบ: {final_code}")
            else:
                st.warning("ไม่พบรหัส")
