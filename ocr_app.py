import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import re
import gc # ช่วยเคลียร์ RAM

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng OCR Scanners", page_icon="⚡", layout="centered")

# ใช้ Try-Except ดักจับ Error ตั้งแต่เริ่มโหลด
try:
    # --- โหลดโมเดล (Cache) ---
    @st.cache_resource
    def load_model():
        # quantize=True ช่วยลดการกิน RAM
        return easyocr.Reader(['en'], gpu=False, quantize=True)

    with st.spinner('กำลังเตรียมระบบ (v5: รองรับรหัส 12 หลัก)...'):
        reader = load_model()

except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดตอนโหลดระบบ: {e}")
    st.stop()

# --- Logic ใหม่: ตรวจสอบเงื่อนไข 12 หลักเท่านั้น ---
def is_valid_pattern(text):
    # เงื่อนไขเดียวคือ ต้องยาว 12 ตัวอักษรเป๊ะๆ
    # (Regex ด้านล่างกรองให้เหลือแค่ A-Z และ 0-9 แล้ว)
    return len(text) == 12

# --- Preprocessing (ปรับปรุงสำหรับการอ่าน 12 หลัก) ---
def enhance_image_for_ocr(image):
    # 1. ลดการตัดขอบลง (จาก 18% เหลือ 12%) 
    # เพราะรหัส 12 หลักจะยาวเกือบถึงขอบฝา ถ้าตัดเยอะจะแหว่ง
    width, height = image.size
    crop_val = 0.12 
    image = image.crop((width*crop_val, height*crop_val, width*(1-crop_val), height*(1-crop_val)))
    
    # 2. Smart Resize (รักษาความกว้างไว้ที่ประมาณ 1000px)
    # ขนาดนี้กำลังดีสำหรับ 12 ตัวอักษร และไม่กิน RAM เกินไป
    target_width = 1000 
    if image.width != target_width:
        w_percent = (target_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((target_width, h_size), resample=Image.LANCZOS)
    
    # 3. ขาวดำ + Equalize + Contrast (สูตรเดิมที่ใช้ได้ดี)
    image = image.convert('L')
    image = ImageOps.equalize(image) # เกลี่ยแสงเงา
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5) # เพิ่ม Contrast พอประมาณ
    
    return image

# --- ฟังก์ชันหลัก (ปรับจูนให้หา 12 หลัก) ---
def smart_read(image_pil):
    try:
        processed_img = enhance_image_for_ocr(image_pil)
        candidates = []

        # หมุน 4 ทิศ (จำเป็นมากสำหรับฝาที่ถ่ายกลับหัว)
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                rotated = processed_img.rotate(-angle, expand=True)
            else:
                rotated = processed_img
                
            img_np = np.array(rotated)
            
            # อ่านค่า (Allowlist A-Z, 0-9)
            results = reader.readtext(img_np, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            full_text = "".join(results).upper()
            # กรองขยะทิ้ง เหลือแค่ตัวอักษรและตัวเลข
            clean_text = re.sub(r'[^A-Z0-9]', '', full_text)
            
            # เคลียร์เมมโมรี่
            del img_np
            del rotated
            gc.collect() 
            
            # --- ค้นหา Pattern 12 ตัวอักษร ---
            # เนื่องจากมันอาจจะอ่านตัวอักษรนูนอื่นๆ ติดมาด้วย
            # เราจะใช้ "หน้าต่าง" ขนาด 12 ตัว เลื่อนหาไปเรื่อยๆ
            if len(clean_text) >= 12:
                for i in range(len(clean_text) - 11):
                    chunk = clean_text[i : i+12]
                    if is_valid_pattern(chunk):
                        # เจอ 12 ตัวเป๊ะๆ ส่งกลับทันที!
                        return chunk

            # ถ้าไม่เจอ 12 เป๊ะๆ ให้เก็บข้อความที่ยาวใกล้เคียงไว้ (10-14 ตัว)
            if len(clean_text) >= 10 and len(clean_text) <= 14:
                candidates.append(clean_text)

        gc.collect() # เคลียร์ RAM ก่อนจบ
        
        # ถ้าวนครบ 4 ทิศแล้วไม่เจอ 12 ตัวเป๊ะๆ เลย
        if candidates:
            # พยายามเลือกตัวที่ใกล้ 12 ตัวที่สุด
            best_guess = sorted(candidates, key=lambda x: abs(len(x) - 12))[0]
            return best_guess
        
        return None

    except Exception as e:
        return f"Error: {str(e)}"

# --- ส่วนแสดงผล ---
try:
    # โหลดโลโก้
    try:
        st.image("banner.png", width=150)
    except:
        pass # ไม่แสดงอะไรถ้าไม่มีไฟล์
        
    st.write("---")
    st.info("ℹ️ รูปแบบใหม่: ค้นหารหัส 12 หลัก (A-Z, 0-9)")

    tab1, tab2 = st.tabs(["📂 อัปโหลดหลายรูป", "📷 ถ่ายรูป"])

    # TAB 1: Batch Upload
    with tab1:
        uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"กำลังประมวลผล {len(uploaded_files)} รูป...")
            st.markdown("---")
            for i, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns([1, 3])
                try:
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, width=100, caption=f"รูปที่ {i+1}")
                    with col2:
                        with st.spinner('Scanning...'):
                            final_code = smart_read(image)
                            if final_code and "Error:" in final_code:
                                st.error(final_code)
                            elif final_code:
                                st.code(final_code, language=None)
                                if len(final_code) == 12:
                                    st.caption("✅ ครบ 12 หลัก")
                                else:
                                    st.caption(f"⚠️ ตรวจสอบ: อ่านได้ {len(final_code)} หลัก (ควรเป็น 12)")
                            else:
                                st.error("❌ อ่านไม่ออก")
                except Exception as e:
                    st.error(f"ไฟล์เสียหาย: {e}")
                st.markdown("---")

    # TAB 2: Camera
    with tab2:
        camera_image = st.camera_input("ถ่ายรูป")
        if camera_image is not None:
            image = Image.open(camera_image)
            with st.spinner('Scanning...'):
                final_code = smart_read(image)
                if final_code and "Error:" in final_code:
                    st.error(final_code)
                elif final_code:
                    st.code(final_code, language=None)
                    if len(final_code) == 12:
                        st.caption("✅ ครบ 12 หลัก")
                    else:
                        st.caption(f"⚠️ ตรวจสอบ: อ่านได้ {len(final_code)} หลัก (ควรเป็น 12)")
                else:
                    st.warning("ไม่พบรหัส")

except Exception as main_e:
    st.error(f"Critical Error: {main_e}")
