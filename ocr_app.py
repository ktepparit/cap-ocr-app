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

    with st.spinner('กำลังโหลดระบบ V7 (Line-by-Line Intelligence)...'):
        reader = load_model()

except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- Logic: ตรวจสอบ 12 หลัก (A-Z, 0-9) ---
def clean_and_check(text):
    # กรองเฉพาะ A-Z และ 0-9
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    # เช็คความยาว
    if len(cleaned) == 12:
        return cleaned, True
    return cleaned, False

# --- Preprocessing (ปรับแสงให้ชัด แต่ไม่ตัดภาพเยอะ) ---
def process_image(image):
    # 1. Resize เป็น 1200px (ความละเอียดกำลังดีสำหรับอ่าน Text)
    target_width = 1200
    if image.width != target_width:
        w_percent = (target_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((target_width, h_size), resample=Image.LANCZOS)
    
    # 2. ตัดขอบทิ้งแค่นิดเดียว (5%) เพื่อกำจัดขอบฝาส่วนโค้ง
    w, h = image.size
    crop_margin = 0.05
    image = image.crop((w*crop_margin, h*crop_margin, w*(1-crop_margin), h*(1-crop_margin)))
    
    # 3. แปลงเป็นขาวดำ + Equalize (ช่วยกู้รายละเอียดในเงา)
    image = image.convert('L')
    image = ImageOps.equalize(image)
    
    # 4. เพิ่ม Contrast (1.5 เท่า) เพื่อให้ตัวหนังสือเด้งออกมา
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    return image

# --- ฟังก์ชันหลัก ---
def smart_read(image_pil):
    try:
        processed_img = process_image(image_pil)
        
        # แสดงภาพที่ระบบเห็น (เพื่อการ Debug)
        st.image(processed_img, caption="ภาพที่ AI มองเห็น (Processed)", width=200)

        candidates = []

        # วนลูปหมุน 4 ทิศ
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                rotated = processed_img.rotate(-angle, expand=True, fillcolor=128)
            else:
                rotated = processed_img
                
            img_np = np.array(rotated)
            
            # อ่านค่าแยกบรรทัด (detail=0 จะได้เป็น List ของข้อความ)
            # ไม่ใช้ allowlist ตรงนี้ เพราะอยากรู้ว่ามันอ่านอะไรออกมาบ้างก่อนคัดกรอง
            results = reader.readtext(img_np, detail=0)
            
            # --- กลยุทธ์ใหม่: ตรวจสอบทีละบรรทัด (Line-by-Line) ---
            found_perfect_match = False
            for line in results:
                cleaned_line, is_12_chars = clean_and_check(line)
                
                # ถ้าเจอบรรทัดที่มี 12 ตัวเป๊ะๆ (เช่น KY7KLWX6RM46) เอาเลย!
                # บรรทัดที่เป็น "P Bev" (4 ตัว) หรือ "21" (2 ตัว) จะถูกปัดตกไปตรงนี้
                if is_12_chars:
                    del img_np, rotated
                    gc.collect()
                    return cleaned_line # เจอ Jackpot จบงานทันที

                # เก็บตัวเลือกที่ใกล้เคียงไว้ (10-14 ตัว) เผื่อไม่มีอันไหนเป๊ะ
                if 10 <= len(cleaned_line) <= 14:
                    candidates.append(cleaned_line)
            
            # ถ้าไม่เจอบรรทัดเดียวเป๊ะๆ ลองเอาทุกบรรทัดมาต่อกัน (เผื่อมันอ่านขาดตอน)
            full_text_joined = "".join(results)
            cleaned_joined, _ = clean_and_check(full_text_joined)
            
            # Sliding Window หา 12 ตัวในข้อความยาวๆ
            if len(cleaned_joined) >= 12:
                for i in range(len(cleaned_joined) - 11):
                    chunk = cleaned_joined[i : i+12]
                    # กรองว่าเป็น A-Z0-9 ล้วนๆ
                    if len(chunk) == 12:
                         # เก็บไว้เป็น candidate แบบความหวังสุดท้าย
                         candidates.append(chunk)

            del img_np, rotated
            gc.collect()

        # สรุปผล: ถ้าไม่เจอเป๊ะๆ ให้เอาตัวที่ดีที่สุดใน Candidates
        if candidates:
            # เรียงลำดับ หาตัวที่ใกล้ 12 ตัวที่สุด
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
    st.info("ℹ️ V7: ระบบแยกบรรทัดอัจฉริยะ (แยก P Bev ออกจากรหัส)")

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
                                    st.caption(f"⚠️ ได้ {len(final_code)} หลัก: {final_code}")
                            elif final_code:
                                st.error(final_code)
                            else:
                                st.error("❌ ไม่พบรหัส (ลองปรับแสงหรือมุมกล้อง)")
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
