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

    with st.spinner('กำลังโหลดระบบ V9 (Fix W, M, 6)...'):
        reader = load_model()

except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- Preprocessing สูตรแก้ตัวบาง/เส้นขาด ---
def process_image(image):
    # 1. Resize ใหญ่ขึ้นอีกนิด (1500px) เพื่อให้เห็นรายละเอียดเส้น
    target_width = 1500
    if image.width != target_width:
        w_percent = (target_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((target_width, h_size), resample=Image.LANCZOS)
    
    # 2. ตัดขอบทิ้ง 10% (ปลอดภัยกว่า 5% เล็กน้อย ตัดขอบโค้งทิ้ง)
    w, h = image.size
    crop_margin = 0.10
    image = image.crop((w*crop_margin, h*crop_margin, w*(1-crop_margin), h*(1-crop_margin)))
    
    # 3. แปลงเป็นขาวดำ
    image = image.convert('L')
    
    # 4. [เคล็ดลับสำคัญ] Gaussian Blur เล็กน้อย
    # ช่วย "เชื่อม" เส้นตัว W และ M ที่ขาดๆ ให้ติดกัน
    image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # 5. Binarization (แปลงเป็น ขาว-ดำ สนิท ไม่มีสีเทา)
    # ตัดที่ค่าความสว่าง 135 (ปรับให้เส้นหนาขึ้น)
    # ตัวเลข 6 จะถูกถมดำจนหัวปิดสนิท แยกออกจาก G ได้
    fn = lambda x : 255 if x > 135 else 0
    image = image.point(fn, mode='1')
    
    return image

# --- ฟังก์ชันหลัก ---
def smart_read(image_pil):
    try:
        processed_img = process_image(image_pil)
        
        # แสดงภาพที่ใช้ประมวลผล (เพื่อดูว่า W ติดกันหรือยัง)
        st.image(processed_img, caption="ภาพที่ AI เห็น (ขาว-ดำ สนิท)", width=200)

        candidates = []

        # วนลูปหมุน 4 ทิศ
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                # convert('L') กลับมาเป็นเทาเพื่อให้หมุนได้เนียนขึ้น
                rotated = processed_img.convert('L').rotate(-angle, expand=True, fillcolor=255)
            else:
                rotated = processed_img.convert('L')
                
            img_np = np.array(rotated)
            
            # อ่านค่า
            results = reader.readtext(img_np, detail=0)
            
            for line_text in results:
                # Cleaning: เก็บ A-Z, 0-9
                cleaned_line = re.sub(r'[^A-Z0-9]', '', line_text.upper())
                
                # --- Correction Logic (แก้คำผิดเฉพาะหน้า) ---
                # ถ้าความยาว 12 ตัว เราจะลองเช็คดูว่ามีตัวที่ AI สับสนบ่อยๆ ไหม
                if len(cleaned_line) == 12:
                    # (Optional) ตรงนี้ถ้าต้องการ Hard code แก้ผิดเป็นถูกทำได้
                    # แต่เราเน้นแก้ที่ภาพก่อน
                    del img_np, rotated
                    gc.collect()
                    return cleaned_line

                # เก็บตัวเลือก
                if 10 <= len(cleaned_line) <= 14:
                    candidates.append(cleaned_line)

            del img_np, rotated
            gc.collect()

        # ถ้าหาเป๊ะๆ ไม่เจอ ให้หาตัวที่ใกล้เคียงที่สุดจาก Candidates
        if candidates:
            # เรียงลำดับ เอาตัวที่ใกล้ 12 หลักที่สุด
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
    st.info("ℹ️ V9: เชื่อมเส้นตัวอักษร (แก้ W, M, 6)")

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
