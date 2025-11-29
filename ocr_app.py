import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import re
import gc

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng OCR Scanners", page_icon="⚡", layout="centered")

try:
    # --- โหลดโมเดล (Cache) ---
    @st.cache_resource
    def load_model():
        # quantize=True ช่วยลด RAM
        return easyocr.Reader(['en'], gpu=False, quantize=True)

    with st.spinner('กำลังโหลดระบบ V10 (Thicken Lines & Best Effort)...'):
        reader = load_model()

except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- Preprocessing สูตร "เติมเนื้อหมึก" ---
def process_image(image):
    # 1. Resize ให้ใหญ่พอดี (1600px)
    target_width = 1600
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
    
    # 4. [ไม้ตาย] MinFilter(3) = ทำให้สีดำหนาขึ้น!
    # ฟิลเตอร์นี้จะเลือกพิกเซลที่มืดที่สุดในรอบๆ 3px มาขยายผล
    # ผลลัพธ์: จุดไข่ปลาจะเชื่อมกัน, เส้นบางๆ จะหนาขึ้น (แก้ W เป็น I)
    image = image.filter(ImageFilter.MinFilter(3))
    
    # 5. เพิ่ม Contrast ให้ชัดเจน
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    return image

# --- ฟังก์ชันหลัก ---
def smart_read(image_pil):
    try:
        processed_img = process_image(image_pil)
        
        # แสดงภาพที่ AI เห็น (สังเกตว่าตัวหนังสือจะดูตัวหนาขึ้นมาก)
        st.image(processed_img, caption="ภาพที่ AI เห็น (เส้นหนาขึ้น)", width=200)

        candidates = []

        # วนลูปหมุน 4 ทิศ
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                rotated = processed_img.rotate(-angle, expand=True, fillcolor=255)
            else:
                rotated = processed_img
                
            img_np = np.array(rotated)
            
            # อ่านค่า
            results = reader.readtext(img_np, detail=0)
            
            for line_text in results:
                # Cleaning: เก็บ A-Z, 0-9
                cleaned_line = re.sub(r'[^A-Z0-9]', '', line_text.upper())
                
                # ถ้าเจอ 12 ตัวเป๊ะๆ ส่งกลับทันที (Jackpot!)
                if len(cleaned_line) == 12:
                    del img_np, rotated
                    gc.collect()
                    return cleaned_line

                # เก็บตัวเลือกที่ "เข้าข่าย" ไว้ (8-15 ตัว)
                # รอบนี้เราเก็บช่วงกว้างขึ้น เผื่อมันอ่านเกินหรือขาดไปบ้าง
                if 8 <= len(cleaned_line) <= 15:
                    candidates.append(cleaned_line)

            del img_np, rotated
            gc.collect()

        # สรุปผล: ถ้าไม่เจอเป๊ะๆ ให้เอาตัวที่ดีที่สุดออกมา (Best Guess)
        if candidates:
            # 1. เรียงลำดับตามความยาว (เอาที่ใกล้ 12 ที่สุด)
            candidates.sort(key=lambda x: abs(len(x) - 12))
            
            # ส่งตัวแรกที่ใกล้เคียงที่สุดกลับไป ดีกว่าไม่ส่งอะไรเลย
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
    st.info("ℹ️ V10: โหมดเส้นหนา + พยายามอ่านให้ได้ (Best Effort)")

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
                                    st.caption(f"⚠️ ได้ {len(final_code)} หลัก (ใกล้เคียงที่สุด)")
                            elif final_code:
                                st.error(final_code)
                            else:
                                st.warning("❌ ไม่พบข้อความที่อ่านออกได้เลย")
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
                        st.caption(f"⚠️ ได้ {len(final_code)} หลัก (ใกล้เคียงที่สุด)")
                else:
                    st.warning("❌ ไม่พบรหัส")

except Exception as main_e:
    st.error(f"Critical: {main_e}")
