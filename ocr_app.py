import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng AI Scanner", page_icon="⚡", layout="centered")

# --- ส่วนใส่ API Key ---
with st.sidebar:
    st.header("🔑 ตั้งค่าระบบ")
    st.success("Model: gemini-2.5-pro") # ใช้ตัว Pro รุ่นฉลาดสุด
    
    default_api_key = "AIzaSyCmWmCTFIZ31hNPYdQMjwGfEzP9SxJnl6o" 
    api_key_input = st.text_input("ใส่ Google API Key", value=default_api_key, type="password")
    api_key = api_key_input if api_key_input else default_api_key
    
    if not api_key:
        st.warning("⚠️ ต้องใส่ API Key ก่อนใช้งาน")

# --- ฟังก์ชันอ่านภาพด้วย Gemini ---
def gemini_vision_scan(image_pil, key):
    try:
        # ตั้งค่าโมเดล
        genai.configure(api_key=key)
        
        # ✅ ใช้โมเดล gemini-2.5-pro ตามที่ขอ
        model = genai.GenerativeModel('gemini-2.5-pro')

        # --- Super Prompt สำหรับรุ่น Pro (สั่งให้คิดวิเคราะห์) ---
        prompt = """
        Analyze the image of the bottle cap to extract the 12-character serial code printed on the inside.
        
        This is a difficult OCR task involving Dot-Matrix fonts. You must use your advanced reasoning to correct common OCR errors based on the context of the alphanumeric code.

        CRITICAL CORRECTION RULES:
        1. **'7' vs 'Z':** The character '7' often has a hooked top in this font, which makes it look like 'Z'. Unless it is unmistakably 'Z', interpret it as '7'.
        2. **'6' vs 'G':** The number '6' often has a gap, looking like 'G'. Check the curvature carefully.
        3. **'W' vs 'I' or 'U':** The letter 'W' is composed of faint dots and can look like 'I', 'U', or 'V'. Look for the width and the faint center dots.
        4. **'M' vs 'H':** Similar to 'W', look for the faint center V-shape of 'M'.
        
        REQUIREMENTS:
        - The code is EXACTLY 12 alphanumeric characters (A-Z, 0-9).
        - Ignore text like "P Bev", "21", "HDPE", "07", or recycling symbols.
        - Do not include spaces or labels.

        OUTPUT:
        Return ONLY the 12-character code string.
        """

        # ส่งรูปและคำสั่งไป
        response = model.generate_content([prompt, image_pil])
        return response.text.strip()
        
    except Exception as e:
        return f"Error: {str(e)}"

# --- ส่วนแสดงผล UI ---
try:
    try:
        st.image("banner.png", width=150)
    except:
        pass 
        
    st.title("⚡ Kratingdaeng AI Scanner")
    st.caption("Powered by: Gemini 2.5 Pro (High Reasoning) 🧠") 
    st.write("---")

    if api_key:
        tab1, tab2 = st.tabs(["📂 อัปโหลดรูป", "📷 ถ่ายรูป"])

        # TAB 1: Upload
        with tab1:
            uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            if uploaded_files:
                st.success(f"ส่งให้ AI (2.5 Pro) วิเคราะห์ {len(uploaded_files)} รูป...")
                st.markdown("---")
                for i, uploaded_file in enumerate(uploaded_files):
                    col1, col2 = st.columns([1, 3])
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, width=100, caption=f"Img {i+1}")
                    with col2:
                        with st.spinner('AI กำลังใช้ความคิด (Pro)...'):
                            code = gemini_vision_scan(image, api_key)
                            
                            if "Error" in code:
                                st.error(code)
                                if "429" in code:
                                    st.warning("⚠️ โมเดล Pro โควต้าเต็มเร็วมากครับ ถ้าใช้ต่อไม่ได้ ให้ลองกลับไปใช้ 'gemini-1.5-flash'")
                            else:
                                clean_code = code.replace(" ", "").replace("\n", "")
                                st.code(clean_code, language=None)
                                
                                if len(clean_code) == 12:
                                    st.caption("✅ ครบ 12 หลัก")
                                else:
                                    st.caption(f"⚠️ อ่านได้ {len(clean_code)} หลัก")
                    st.markdown("---")

        # TAB 2: Camera
        with tab2:
            camera_image = st.camera_input("ถ่ายรูป")
            if camera_image is not None:
                image = Image.open(camera_image)
                with st.spinner('AI กำลังใช้ความคิด (Pro)...'):
                    code = gemini_vision_scan(image, api_key)
                    if "Error" in code:
                        st.error(code)
                    else:
                        clean_code = code.replace(" ", "").replace("\n", "")
                        st.code(clean_code, language=None)
                        if len(clean_code) == 12:
                            st.caption("✅ ครบ 12 หลัก")
                        else:
                            st.caption(f"⚠️ อ่านได้ {len(clean_code)} หลัก")
    else:
        st.info("👈 กรุณาใส่ API Key ทางด้านซ้ายเพื่อเริ่มใช้งาน")

except Exception as main_e:
    st.error(f"Critical: {main_e}")
