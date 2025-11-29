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
    # allowlist: อ่านเฉพาะ A-Z และ 0-9
    return easyocr.Reader(['en'], gpu=False)

with st.spinner('กำลังเตรียมระบบอ่านรหัส...'):
    reader = load_model()

# --- ฟังก์ชันช่วย: ตรวจสอบกฎเหล็ก (Logic ตามที่คุณขอ) ---
def is_valid_pattern(text):
    # 1. ต้องยาว 9 ตัว
    if len(text) != 9:
        return False
        
    # 2. นับจำนวนตัวเลข
    digit_count = sum(c.isdigit() for c in text)
    
    # เงื่อนไข: (ตัวเลข 0 ตัว) หรือ (ตัวเลข 2 ตัว) เท่านั้น
    if digit_count == 0 or digit_count == 2:
        return True
    
    return False

# --- ฟังก์ชันช่วย: ตัดขอบภาพ (Center Crop) ---
def crop_center(image, crop_percent=15):
    # ตัดขอบออกด้านละ 15-20% เพื่อตัดคำว่า HDPE หรือตัวเลขนูนที่ขอบฝาออก
    width, height = image.size
    left = (width * crop_percent) / 100
    top = (height * crop_percent) / 100
    right = width - left
    bottom = height - top
    return image.crop((left, top, right, bottom))

# --- ฟังก์ชันหลัก: อ่านและคัดกรอง ---
def smart_read(image_pil):
    # 1. ตัดขอบทิ้งก่อนเลย (กำจัด HDPE, PAT, etc.)
    img_cropped = crop_center(image_pil, crop_percent=18) 
    
    # 2. ทำภาพให้เป็นขาว-ดำ (Binarization) แบบเข้มข้น
    # เปลี่ยนเป็น Grayscale
    img_gray = img_cropped.convert('L')
    
    # เร่ง Contrast สูงมาก
    enhancer = ImageEnhance.Contrast(img_gray)
    img_high_contrast = enhancer.enhance(3.0) 
    
    candidates = []

    # 3. วนลูปหมุนภาพ 4 ทิศ (0, 90, 180, 270)
    for angle in [0, 90, 180, 270]:
        # หมุนภาพ
        if angle != 0:
            rotated_img = img_high_contrast.rotate(-angle, expand=True)
        else:
            rotated_img = img_high_contrast
        
        # แปลงเป็น numpy
        img_np = np.array(rotated_img)
        
        # อ่านค่า (Allowlist A-Z 0-9)
        results = reader.readtext(img_np, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        # รวม text ที่อ่านได้ในมุมนั้นๆ
        full_text_raw = "".join(results)
        full_text_upper = full_text_raw.upper()
        
        # กรองเอาเฉพาะ A-Z และ 0-9
        clean_text = re.sub(r'[^A-Z0-9]', '', full_text_upper)
        
        # --- ตรวจสอบ Logic ---
        # ลองหาช่วงที่มี 9 ตัวอักษรเรียงกัน
        # เนื่องจากบางทีมันอ่านขยะมาติดด้วย เราจะ slide window หา 9 ตัว
        for i in range(len(clean_text) - 8):
            chunk = clean_text[i : i+9]
            if is_valid_pattern(chunk):
                return chunk # เจอ Pattern ที่ใช่ (ยาว 9, เลข 0 หรือ 2 ตัว) ส่งกลับทันที!

        # กรณีไม่เจอ 9 ตัวเป๊ะๆ แต่เจอ text ยาวๆ เก็บไว้เป็นตัวสำรอง
        if len(clean_text) >= 8 and len(clean_text) <= 12:
            candidates.append(clean_text)

    # ถ้าไม่เจอ Pattern เทพ (9 ตัวเป๊ะ) ให้พยายามเลือกตัวที่ดีที่สุดจากตัวสำรอง
    if candidates:
        # ลองวนหาตัวที่มีเลข 2 ตัว หรือ 0 ตัว ในบรรดาตัวสำรอง
        for cand in candidates:
             d_count = sum(c.isdigit() for c in cand)
             if d_count == 2 or d_count == 0:
                 return cand
        
        # ถ้าไม่มีจริงๆ เอาตัวที่ยาวใกล้เคียง 9 สุด
        return max(candidates, key=len)
    
    return None

# --- ส่วนแสดงผลโลโก้ (Fix ขนาด 150x201) ---
try:
    logo = Image.open("banner.png")
    logo_resized = logo.resize((150, 201))
    col_logo, col_space = st.columns([1, 2])
    with col_logo:
        st.image(logo_resized)
except FileNotFoundError:
    st.warning("ไม่พบไฟล์ banner.png")

# --- ส่วนเนื้อหาหลัก ---
st.write("---")
st.info("ℹ️ กฎเหล็ก: รหัส 9 หลัก (มีตัวเลขได้เพียง 0 หรือ 2 ตัวเท่านั้น) + ตัดขอบรบกวนอัตโนมัติ")

tab1, tab2 = st.tabs(["📂 อัปโหลดหลายรูป", "📷 ถ่ายรูป"])

# ================= TAB 1: Batch Upload =================
with tab1:
    uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"กำลังสแกน {len(uploaded_files)} รูป...")
        st.markdown("---")
        
        for i, uploaded_file in enumerate(uploaded_files):
            col1, col2 = st.columns([1, 3])
            image = Image.open(uploaded_file)

            with col1:
                st.image(image, width=100, caption=f"รูปที่ {i+1}")

            with col2:
                with st.spinner('กำลังวิเคราะห์...'):
                    final_code = smart_read(image)
                    
                    if final_code:
                        st.code(final_code, language=None)
                        
                        # ตรวจสอบความถูกต้องเพื่อแสดงสีสถานะ
                        d_c = sum(c.isdigit() for c in final_code)
                        if len(final_code) == 9 and (d_c == 0 or d_c == 2):
                            st.caption("✅ ผ่านเงื่อนไข (9 หลัก, เลข 0/2 ตัว)")
                        else:
                            st.caption(f"⚠️ รูปแบบไม่ตรงเงื่อนไข 100% (ยาว {len(final_code)}, เลข {d_c} ตัว)")
                    else:
                        st.error("❌ อ่านค่าไม่ได้")
            st.markdown("---")

# ================= TAB 2: Camera =================
with tab2:
    camera_image = st.camera_input("ถ่ายรูปฝาขวด")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.write("ผลลัพธ์:")
        with st.spinner('กำลังวิเคราะห์...'):
            final_code = smart_read(image)
            if final_code:
                st.code(final_code, language=None)
                d_c = sum(c.isdigit() for c in final_code)
                if len(final_code) == 9 and (d_c == 0 or d_c == 2):
                    st.caption("✅ ผ่านเงื่อนไข")
                else:
                    st.caption(f"⚠️ ตรวจสอบอีกครั้ง (ยาว {len(final_code)}, เลข {d_c} ตัว)")
            else:
                st.warning("ไม่พบรหัส")
