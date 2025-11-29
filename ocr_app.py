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
        return easyocr.Reader(['en'], gpu=False, quantize=True)

    with st.spinner('กำลังเตรียมระบบ (V6: Focus Middle Strip)...'):
        reader = load_model()

except Exception as e:
    st.error(f"❌ System Load Error: {e}")
    st.stop()

# --- Logic: ตรวจสอบ 12 หลัก ---
def is_valid_pattern(text):
    return len(text) == 12

# --- Preprocessing (สูตรใหม่: เจาะจงพื้นที่ตรงกลาง) ---
def process_image(image):
    # 1. Resize เป็น 1200px (ความละเอียดสูงขึ้นเพื่อให้อ่าน W ชัดๆ)
    target_width = 1200
    if image.width != target_width:
        w_percent = (target_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((target_width, h_size), resample=Image.LANCZOS)
    
    # 2. [สำคัญมาก] ตัดส่วนบนและล่างทิ้งเยอะๆ (Vertical Crop)
    # ตัดบน 28% และล่าง 28% ทิ้ง -> เหลือพื้นที่ตรงกลางแค่ 44%
    # วิธีนี้จะกำจัดคำว่า "P Bev" (ด้านบน) และตัวเลขนูน (ด้านล่าง) ออกไปเลย
    w, h = image.size
    top_crop = h * 0.28
    bottom_crop = h * 0.72 # (100% - 28%)
    
    # ตัดซ้ายขวานิดหน่อย (10%)
    left_crop = w * 0.10
    right_crop = w * 0.90
    
    image = image.crop((left_crop, top_crop, right_crop, bottom_crop))
    
    # 3. ขาวดำ + เร่ง Contrast จัดๆ
    image = image.convert('L')
    
    # ใช้ UnsharpMask เพื่อเน้นขอบตัวหนังสือให้คมกริบ (แก้ W อ่านผิด)
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
    
    # เร่งความสว่างและ Contrast ให้พื้นหลังหายไป
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(3.0) # เร่ง 3 เท่า
    
    # เพิ่มความเข้ม (Thresholding แบบบ้านๆ) เพื่อให้เส้นบางๆ ของตัว W ชัดขึ้น
    # โดยการทำให้ส่วนที่ไม่ใช่สีขาว กลายเป็นดำให้หมด
    image = image.point(lambda p: p if p > 160 else 0)
    
    return image

# --- ฟังก์ชันหลัก ---
def smart_read(image_pil):
    try:
        processed_img = process_image(image_pil)
        candidates = []

        # วนลูปหมุน 4 ทิศ (สำคัญมากสำหรับฝากลับหัว)
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                rotated = processed_img.rotate(-angle, expand=True, fillcolor=255)
            else:
                rotated = processed_img
                
            img_np = np.array(rotated)
            
            # อ่านค่า
            # paragraph=True อาจช่วยรวมคำที่ขาดตอนได้
            results = reader.readtext(img_np, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            full_text = "".join(results).upper()
            clean_text = re.sub(r'[^A-Z0-9]', '', full_text)
            
            del img_np, rotated
            gc.collect() 
            
            # 1. หา 12 ตัวเป๊ะ
            if len(clean_text) >= 12:
                # Sliding Window หาช่วงที่ดีที่สุด
                for i in range(len(clean_text) - 11):
                    chunk = clean_text[i : i+12]
                    # กรองเบื้องต้น: รหัสที่ดีมักจะไม่มีตัวเลขติดกันยาวเหยียดเกินไป (Optional logic)
                    if is_valid_pattern(chunk):
                        return chunk

            # เก็บไว้เป็นตัวเลือก
            if len(clean_text) >= 10 and len(clean_text) <= 15:
                candidates.append(clean_text)

        gc.collect()
        
        # ถ้าหาเป๊ะๆ ไม่เจอ ให้เอาตัวที่ยาว 12 หรือใกล้เคียงที่สุด
        if candidates:
            best_guess = sorted(candidates, key=lambda x: abs(len(x) - 12))[0]
            return best_guess
        
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
    st.info("ℹ️ V6: Focus Middle Strip (ตัดรอยนูนบนล่างทิ้ง)")

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
                        st.image(image, width=100, caption=f"Img {i+1}")
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
                                st.error("❌ ไม่พบรหัส")
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
