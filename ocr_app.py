import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng AI Scanner", page_icon="⚡", layout="centered")

# --- ส่วนใส่ API Key ---
with st.sidebar:
    st.header("🔑 ตั้งค่าระบบ")
    st.success("Model: gemini-2.5-flash") # เปลี่ยนมาใช้ตัว Flash ปกติ
    
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
        
        # ✅ เปลี่ยนมาใช้ gemini-2.5-flash ตามที่ขอ
        model = genai.GenerativeModel('gemini-2.5-flash')

        # --- Prompt ที่ยังคงความฉลาดในการแก้คำผิด ---
        prompt = """
        Analyze the image of the bottle cap to extract the 12-character serial code.
        
        CRITICAL INSTRUCTIONS FOR DOT-MATRIX FONT:
        1. **Context:** The text is printed with dots. Connections might be faint.
        2. **Common Errors to Fix:**
           - **'7' vs 'Z':** The number '7' often looks like 'Z' in this font. If in doubt, choose '7'.
           - **'6' vs 'G':** The number '6' often looks like 'G'. Check if the loop is closed.
           - **'W' vs 'I':** 'W' is wide. Do not mistake it for 'I'.
           - **'M' vs 'H':** Check the middle part of 'M'.
        
        3. **Format:** The code is EXACTLY 12 alphanumeric characters (A-Z, 0-9).
        4. **Ignore:** "P Bev", "21", "HDPE", recycling symbols.

        OUTPUT: Return ONLY the 12-character code text.
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
    st.caption("Powered by: Gemini 2.5 Flash ⚡") 
    st.write("---")

    if api_key:
        tab1, tab2 = st.tabs(["📂 อัปโหลดรูป", "📷 ถ่ายรูป"])

        # TAB 1: Upload
        with tab1:
            uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            if uploaded_files:
                st.success(f"ส่งให้ AI (2.5 Flash) วิเคราะห์ {len(uploaded_files)} รูป...")
                st.markdown("---")
                for i, uploaded_file in enumerate(uploaded_files):
                    col1, col2 = st.columns([1, 3])
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, width=100, caption=f"Img {i+1}")
                    with col2:
                        with st.spinner('กำลังอ่าน...'):
                            code = gemini_vision_scan(image, api_key)
                            
                            if "Error" in code:
                                st.error(code)
                                if "429" in code:
                                    st.warning("⚠️ โควต้าเต็มอีกแล้ว ลองเปลี่ยนเป็น 'gemini-1.5-flash' (ตัวเสถียรสุด) แทนครับ")
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
                with st.spinner('กำลังอ่าน...'):
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
