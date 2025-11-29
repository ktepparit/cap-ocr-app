import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import re
import gc

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng OCR Scanners", page_icon="⚡", layout="centered")

try:
    # --- โหลดโมเดล ---
    @st.cache_resource
    def load_model():
        return easyocr.Reader(['en'], gpu=False, quantize=True)

    with st.spinner('กำลังโหลดระบบ V11 (Negative & Sharp)...'):
        reader = load_model()

except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- Preprocessing สูตร "กลับสี + คมชัด" ---
def process_image(image):
    # 1. Resize (1200px กําลังดี ไม่ใหญ่จนภาพแตก)
    target_width = 1200
    if image.width != target_width:
        w_percent = (target_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((target_width, h_size), resample=Image.LANCZOS)
    
    # 2. ตัดขอบทิ้ง 10%
    w, h = image.size
    crop_margin = 0.10
    image = image.crop((w*crop_margin, h*crop_margin, w*(1-crop_margin), h*(1-crop_margin)))
    
    # 3. แปลงเป็นขาวดำ
    image = image.convert('L')
    
    # 4. [ไม้ตาย V11] กลับสี (Invert)
    # เปลี่ยนตัวหนังสือดำ เป็น "ตัวหนังสือขาวพื้นดำ"
    # ช่วยลดแสงสะท้อน และทำให้จุดไข่ปลาเด่นขึ้น
    image = ImageOps.invert(image)
    
    # 5. ปรับแสงอัตโนมัติ (Auto Contrast)
    image = ImageOps.autocontrast(image, cutoff=2)
    
    # 6. เพิ่มความคมชัด (Sharpen) 
    # ไม่ใช้ Blur แล้ว เพราะทำให้ 7 เป็น Z
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # 7. เร่ง Contrast อีกนิด
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    return image

# --- ฟังก์ชันแก้คำผิด (Heuristic Correction) ---
def autocorrect(text):
    # ถ้ามั่นใจว่าเป็นรหัส 12 หลัก แต่มีตัวอักษรที่มักอ่านผิด
    # เราสามารถสร้างกฎการแทนที่ได้ (ใช้ด้วยความระมัดระวัง)
    corrected = list(text)
    
    # กฎนี้จะทำงานเมื่อความยาวครบ 12 เท่านั้น
    # ตัวอย่าง: Z มักจะเป็นเลข 7 ในฟอนต์นี้
    # ตัวอย่าง: U มักจะเป็น W (ถ้าอยู่ในบริบทนี้)
    # แต่ต้องระวังเพราะ Z และ U อาจจะเป็นตัวจริงก็ได้ 
    # *ในเวอร์ชั่นนี้ผมขอไม่บังคับเปลี่ยน แต่ comment ไว้ให้คุณดูแนวทาง*
    
    return text

# --- ฟังก์ชันหลัก ---
def smart_read(image_pil):
    try:
        processed_img = process_image(image_pil)
        
        # แสดงภาพที่ AI เห็น (จะเป็นพื้นดำ ตัวขาว)
        st.image(processed_img, caption="ภาพที่ AI เห็น (Inverted)", width=200)

        candidates = []

        # วนลูปหมุน 4 ทิศ
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                rotated = processed_img.rotate(-angle, expand=True, fillcolor=0) # fillcolor=0 สีดำ
            else:
                rotated = processed_img
                
            img_np = np.array(rotated)
            
            # อ่านค่า
            results = reader.readtext(img_np, detail=0)
            
            for line_text in results:
                # Cleaning
                cleaned_line = re.sub(r'[^A-Z0-9]', '', line_text.upper())
                
                # เก็บ 12 ตัวเป๊ะ
                if len(cleaned_line) == 12:
                    del img_np, rotated
                    gc.collect()
                    return cleaned_line

                # เก็บตัวเลือกใกล้เคียง
                if 10 <= len(cleaned_line) <= 14:
                    candidates.append(cleaned_line)

            del img_np, rotated
            gc.collect()

        # Best Guess
        if candidates:
            # เรียงลำดับเอาใกล้ 12 ที่สุด
            candidates.sort(key=lambda x: abs(len(x) - 12))
            return candidates[0]
        
        return None

    except Exception as e:
        return f"Error: {str(e)}"

# --- UI Display ---
try:
    try:
        st.image("banner.png", width=150)
    except:
        pass 
        
    st.write("---")
    st.info("ℹ️ V11: โหมด Negative Image (ตัวขาวพื้นดำ) เพื่อความคมชัด")

    tab1, tab2 = st.tabs(["📂 อัปโหลดหลายรูป", "📷 ถ่ายรูป"])

    # TAB 1
    with tab1:
        uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"Processing {len(uploaded_files)} images...")
            st.markdown("---")
            for i, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns([1, 3])
                try:
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, width=100, caption=f"Original {i+1}")
                    with col2:
                        with st.spinner('Scanning...'):
                            final_code = smart_read(image)
                            
                            if final_code and "Error" not in final_code:
                                st.code(final_code, language=None)
                                if len(final_code) == 12:
                                    st.caption("✅ ครบ 12 หลัก")
                                else:
                                    st.caption(f"⚠️ ได้ {len(final_code)} หลัก")
                            elif final_code:
                                st.error(final_code)
                            else:
                                st.warning("❌ ไม่พบรหัส")
                except Exception as e:
                    st.error(f"File Error: {e}")
                st.markdown("---")

    # TAB 2
    with tab2:
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            image = Image.open(camera_image)
            with st.spinner('Scanning...'):
                final_code = smart_read(image)
                if final_code and "Error" not in final_code:
                    st.code(final_code, language=None)
                    if len(final_code) == 12:
                        st.caption("✅ ครบ 12 หลัก")
                    else:
                        st.caption(f"⚠️ ได้ {len(final_code)} หลัก")
                else:
                    st.warning("ไม่พบรหัส")

except Exception as main_e:
    st.error(f"Critical: {main_e}")
