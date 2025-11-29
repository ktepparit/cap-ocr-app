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
        # โหลดโมเดลภาษาอังกฤษ
        return easyocr.Reader(['en'], gpu=False, quantize=True)

    with st.spinner('กำลังโหลดระบบ V8 (Strict Single-Line)...'):
        reader = load_model()

except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- Preprocessing (เน้นให้ตัวอักษรคมชัด) ---
def process_image(image):
    # 1. Resize เป็น 1200px
    target_width = 1200
    if image.width != target_width:
        w_percent = (target_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((target_width, h_size), resample=Image.LANCZOS)
    
    # 2. ตัดขอบทิ้งนิดเดียว (5%)
    w, h = image.size
    crop_margin = 0.05
    image = image.crop((w*crop_margin, h*crop_margin, w*(1-crop_margin), h*(1-crop_margin)))
    
    # 3. แปลงเป็นขาวดำ
    image = image.convert('L')
    
    # 4. เพิ่มความคมชัด (Sharpen) ก่อนเร่ง Contrast
    # ช่วยให้ขอบตัว W และเลข 7 ชัดขึ้น
    image = image.filter(ImageFilter.SHARPEN)
    
    # 5. เร่ง Contrast ให้ตัวหนังสือดำเข้มบนพื้นสว่าง
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0) # เพิ่มเป็น 2 เท่า
    
    return image

# --- ฟังก์ชันหลัก (Logic ใหม่: เข้มงวดสุดๆ) ---
def smart_read(image_pil):
    try:
        processed_img = process_image(image_pil)
        
        # แสดงภาพที่ระบบใช้ประมวลผล
        st.image(processed_img, caption="ภาพที่ AI เห็น (V8)", width=200)

        # วนลูปหมุน 4 ทิศ
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                rotated = processed_img.rotate(-angle, expand=True, fillcolor=255)
            else:
                rotated = processed_img
                
            img_np = np.array(rotated)
            
            # อ่านค่าทีละบรรทัด (ได้ผลลัพธ์เป็น List ของข้อความ)
            # detail=0 คือขอแค่ข้อความ ไม่เอาพิกัด
            results = reader.readtext(img_np, detail=0)
            
            # --- V8 Logic: ตรวจสอบทีละบรรทัดอย่างเข้มงวด ---
            for line_text in results:
                # 1. ทำความสะอาด: แปลงเป็นพิมพ์ใหญ่, เก็บเฉพาะ A-Z และ 0-9
                cleaned_line = re.sub(r'[^A-Z0-9]', '', line_text.upper())
                
                # 2. กฎเหล็ก: ต้องมีความยาว 12 ตัวอักษรเป๊ะๆ ในบรรทัดเดียว
                if len(cleaned_line) == 12:
                    # เจอแล้ว! บรรทัดนี้คือรหัสแน่นอน
                    del img_np, rotated
                    gc.collect()
                    return cleaned_line # ส่งคืนค่าทันที ไม่ต้องหาต่อ

            # ถ้าจบลูป results แล้วยังไม่เจอ 12 ตัวเป๊ะในมุมนี้
            # ก็ถือว่ามุมนี้ไม่มีรหัส -> เคลียร์เมมแล้วไปหมุนมุมต่อไป
            del img_np, rotated
            gc.collect()

        # ถ้าวนครบ 4 ทิศแล้วไม่มีบรรทัดไหนผ่านกฎ 12 ตัวเลย
        return None # สรุปว่าหาไม่เจอ

    except Exception as e:
        return f"Error: {str(e)}"

# --- UI Display ---
try:
    try:
        st.image("banner.png", width=150)
    except:
        pass 
        
    st.write("---")
    st.info("ℹ️ V8: โหมดเข้มงวด (อ่านเฉพาะบรรทัดที่มี 12 หลักเท่านั้น)")

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
                                st.caption("✅ พบรหัส 12 หลักในบรรทัดเดียว")
                            elif final_code:
                                st.error(final_code)
                            else:
                                st.warning("❌ ไม่พบรหัสที่ถูกต้องตามเงื่อนไข")
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
                    st.caption("✅ พบรหัส 12 หลักในบรรทัดเดียว")
                else:
                    st.warning("❌ ไม่พบรหัสที่ถูกต้องตามเงื่อนไข")

except Exception as main_e:
    st.error(f"Critical: {main_e}")
