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
        # ใช้ quantize=True เพื่อความเร็วและประหยัด RAM
        return easyocr.Reader(['en'], gpu=False, quantize=True)

    with st.spinner('กำลังโหลดระบบ V12 (Morphological + Auto-Fix)...'):
        reader = load_model()

except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- Preprocessing: Morphological Closing (เชื่อมจุด แต่ไม่บวม) ---
def process_image(image):
    # 1. Resize เป็น 1200px (ความละเอียดที่ EasyOCR ชอบ)
    target_width = 1200
    if image.width != target_width:
        w_percent = (target_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((target_width, h_size), resample=Image.LANCZOS)
    
    # 2. ตัดขอบ 10%
    w, h = image.size
    crop_margin = 0.10
    image = image.crop((w*crop_margin, h*crop_margin, w*(1-crop_margin), h*(1-crop_margin)))
    
    # 3. แปลงเป็นขาวดำ
    image = image.convert('L')
    
    # 4. Invert (กลับสี) -> ตัวหนังสือขาว พื้นดำ
    image = ImageOps.invert(image)
    
    # 5. [ไม้ตาย] Morphological Closing (Dilation -> Erosion)
    # ขั้นที่ 5.1: MaxFilter (Dilation) = ขยายสีขาว (เชื่อมจุด W, M ให้ติดกัน)
    image = image.filter(ImageFilter.MaxFilter(3))
    
    # ขั้นที่ 5.2: MinFilter (Erosion) = หดสีขาวกลับ (คืนรูปทรงเดิม ไม่ให้ 7 กลายเป็น Z)
    image = image.filter(ImageFilter.MinFilter(3))
    
    # 6. Invert กลับคืน (ตัวดำ พื้นขาว)
    image = ImageOps.invert(image)
    
    # 7. เพิ่ม Contrast ปิดท้าย
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    return image

# --- ฟังก์ชันแก้คำผิด (Dictionary Correction) ---
def apply_correction(text):
    # แปลงเป็นตัวพิมพ์ใหญ่ก่อน
    text = text.upper()
    
    # Dictionary สำหรับแก้คำผิดที่พบบ่อยในฟอนต์ Dot Matrix
    # Z -> 7 (เพราะ 7 มักมีหัวงุ้มจนเหมือน Z)
    # G -> 6 (เพราะ 6 หัวไม่ปิดจนเหมือน G)
    # I -> W (ถ้าเจอในตำแหน่งที่ควรเป็น W) - อันนี้แก้ยาก ใช้ Image Process ช่วยแล้ว
    # H -> M (ใช้ Image Process ช่วยแล้ว)
    
    # กฎการแทนที่ (Replace Rules)
    # เราจะแทนที่เฉพาะตัวที่มีโอกาสผิดสูงมากๆ
    text = text.replace('Z', '7')
    # text = text.replace('G', '6') # อันนี้เสี่ยง เพราะ G อาจจะมีจริง
    
    # แต่ถ้าเป็นรหัสที่มั่นใจว่า "ต้องมีตัวเลขน้อย" หรือ "มีรูปแบบเฉพาะ"
    # เราสามารถปรับแก้ตรงนี้ได้
    
    return text

# --- ฟังก์ชันหลัก ---
def smart_read(image_pil):
    try:
        processed_img = process_image(image_pil)
        
        # แสดงภาพที่ AI เห็น
        st.image(processed_img, caption="ภาพที่ AI เห็น (Connected Dots)", width=200)

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
                # Cleaning
                cleaned_line = re.sub(r'[^A-Z0-9]', '', line_text.upper())
                
                # --- Auto Correction ---
                # ลองแก้ Z เป็น 7 ทันที
                corrected_line = apply_correction(cleaned_line)
                
                # ถ้าความยาว 12 เป๊ะ (ทั้งก่อนแก้และหลังแก้)
                if len(corrected_line) == 12:
                    del img_np, rotated
                    gc.collect()
                    return corrected_line

                # เก็บตัวเลือกใกล้เคียง
                if 10 <= len(corrected_line) <= 14:
                    candidates.append(corrected_line)

            del img_np, rotated
            gc.collect()

        # Best Guess
        if candidates:
            # เรียงตามความยาว 12
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
    st.info("ℹ️ V12: เชื่อมจุด (M/W) + แก้ Z เป็น 7 อัตโนมัติ")

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
