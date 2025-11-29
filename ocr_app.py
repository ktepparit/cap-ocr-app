import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import re

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng OCR Scanners", page_icon="⚡", layout="centered")

# --- โหลดโมเดล (Cache) ---
@st.cache_resource
def load_model():
    # โหลดโมเดลโดยระบุ allowlist คืออนุญาตให้อ่านเฉพาะ A-Z และ 0-9 เท่านั้น
    # วิธีนี้ช่วยลดการอ่านผิดเป็นสัญลักษณ์แปลกๆ ได้มาก
    return easyocr.Reader(['en'], gpu=False)

with st.spinner('กำลังเตรียมระบบอ่านรหัส...'):
    reader = load_model()

# --- ฟังก์ชันช่วย: หมุนภาพ ---
def rotate_image(image, angle):
    if angle == 0: return image
    return image.rotate(-angle, expand=True)

# --- ฟังก์ชันช่วย: แก้คำผิดที่พบบ่อย (Common Mistake Fixer) ---
def fix_common_mistakes(text):
    # ตัวอย่าง: บางทีอ่านเลข 0 เป็นตัว O หรือเลข 5 เป็นตัว S 
    # แต่เนื่องจากรหัสมีทั้งตัวเลขและตัวอักษรปนกัน เราจะเน้นแก้เฉพาะที่มั่นใจ
    # หรือถ้าต้องการแก้เฉพาะจุดสามารถใส่ logic เพิ่มตรงนี้ได้
    return text

# --- ฟังก์ชันหลัก: อ่านและคัดกรอง ---
def smart_read(image_pil):
    # 1. Preprocessing: ปรับภาพให้สู้แสงสะท้อนฝาขวด
    # แปลงเป็นขาวดำ (Grayscale)
    img_processed = image_pil.convert('L') 
    
    # เพิ่มความคมชัด (Sharpen) เพื่อให้ขอบตัวหนังสือชัดขึ้น
    img_processed = img_processed.filter(ImageFilter.SHARPEN)
    
    # เพิ่ม Contrast จัดๆ เพื่อแยกสีดำออกจากสีทอง
    enhancer = ImageEnhance.Contrast(img_processed)
    img_processed = enhancer.enhance(2.5) # เพิ่มเป็น 2.5 เท่า
    
    candidates = []

    # วนลูปหมุนภาพ 4 ทิศ (0, 90, 180, 270)
    for angle in [0, 90, 180, 270]:
        rotated_img = rotate_image(img_processed, angle)
        img_np = np.array(rotated_img)
        
        # --- หัวใจสำคัญ: allowlist ---
        # สั่งให้ EasyOCR สนใจแค่ตัวอักษรพิมพ์ใหญ่และตัวเลขเท่านั้น
        results = reader.readtext(img_np, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        full_text = "".join(results)
        
        # คลีนข้อมูลอีกรอบ
        clean_text = re.sub(r'[^A-Z0-9]', '', full_text.upper())
        
        # ถ้าเจอ 9 ตัวเป๊ะๆ ส่งกลับเลย (ถือว่าเจอ Jackpot)
        matches = re.findall(r'[A-Z0-9]{9}', clean_text)
        for match in matches:
            return match 

        # ถ้าไม่เจอเป๊ะ ให้เก็บพวกที่ยาว 8-12 ตัวไว้พิจารณา
        if len(clean_text) >= 8 and len(clean_text) <= 12:
             candidates.append(clean_text)

    # เลือกตัวที่ดีที่สุด (ยาวที่สุด หรือ ใกล้เคียง 9 ที่สุด)
    if candidates:
        # เรียงลำดับเอาตัวที่ความยาวใกล้เลข 9 มากที่สุด
        best_candidate = sorted(candidates, key=lambda x: abs(len(x) - 9))[0]
        return best_candidate
    
    return None

# --- ส่วนแสดงผลโลโก้ (ปรับขนาด 150x201) ---
try:
    logo = Image.open("banner.png")
    # ปรับขนาดภาพให้เป็น 150x201 px ตามที่ขอ
    logo_resized = logo.resize((150, 201))
    
    # สร้าง 3 คอลัมน์เพื่อจัดให้โลโก้อยู่ตรงกลาง (หรือชิดซ้ายตาม default)
    col_logo, col_space = st.columns([1, 2]) # ปรับอัตราส่วนถ้าต้องการจัดกลาง
    with col_logo:
        st.image(logo_resized)
        
except FileNotFoundError:
    st.warning("ไม่พบไฟล์ banner.png กรุณาอัปโหลดรูปภาพ")
    st.title("⚡ ระบบสแกนรหัส")

# --- ส่วนเนื้อหาหลัก ---
st.write("---")
st.subheader("ระบบอ่านรหัส (High Precision Mode)")

tab1, tab2 = st.tabs(["📂 อัปโหลดหลายรูป", "📷 ถ่ายรูป"])

# ================= TAB 1: Batch Upload =================
with tab1:
    uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"กำลังสแกน {len(uploaded_files)} รูป (โหมดความแม่นยำสูง)...")
        st.markdown("---")
        
        for i, uploaded_file in enumerate(uploaded_files):
            col1, col2 = st.columns([1, 3])
            image = Image.open(uploaded_file)

            with col1:
                st.image(image, width=100, caption=f"รูปที่ {i+1}")

            with col2:
                with st.spinner('...'):
                    final_code = smart_read(image)
                    
                    if final_code:
                        st.code(final_code, language=None)
                        if len(final_code) == 9:
                            st.caption("✅ ครบ 9 หลัก")
                        else:
                            st.caption(f"⚠️ อ่านได้ {len(final_code)} หลัก (ตรวจสอบอีกครั้ง)")
                    else:
                        st.error("❌ ไม่พบรหัส")
            st.markdown("---")

# ================= TAB 2: Camera =================
with tab2:
    camera_image = st.camera_input("ถ่ายรูปฝาขวด")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.write("ผลลัพธ์:")
        with st.spinner('กำลังสแกน...'):
            final_code = smart_read(image)
            if final_code:
                st.code(final_code, language=None)
                if len(final_code) == 9:
                    st.caption("✅ ครบ 9 หลัก")
                else:
                     st.caption(f"⚠️ อ่านได้ {len(final_code)} หลัก")
            else:
                st.warning("ไม่พบรหัสที่ชัดเจน")
