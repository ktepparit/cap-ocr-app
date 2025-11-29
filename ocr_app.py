import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng AI Scanner", page_icon="⚡", layout="centered")

# --- ส่วนใส่ API Key ---
with st.sidebar:
    st.header("🔑 ตั้งค่าระบบ")
    st.success("Model: gemini-2.0-flash")
    
    # ⚠️ ใส่ API Key ของคุณตรงนี้ (หรือจะกรอกหน้าเว็บก็ได้)
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
        
        # ✅ เปลี่ยนมาใช้ตัว 2.0 Flash ตามที่ขอ (เสถียรและฟรี)
        model = genai.GenerativeModel('gemini-2.0-flash')

        # --- Prompt คำสั่ง (สั่งให้ AI เพ่งเล็งเฉพาะรหัส) ---
        prompt = """
        You are an expert OCR system for reading codes on bottle caps.
        
        Task: Identify the 12-character alphanumeric serial code printed on the inside of the cap.

        Rules:
        1. The code is EXACTLY 12 characters long.
        2. Characters are uppercase A-Z and 0-9.
        3. IGNORE interference text such as "P Bev", "21", "HDPE", recycling logos, or small mold numbers.
        4. Handle dot-matrix fonts intelligently (e.g., differentiate 'Z' from '7', 'W' from 'I', 'G' from '6' based on context).
        
        Output: Just the 12-character code. Nothing else.
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
    st.caption("Powered by: Gemini 2.0 Flash") 
    st.write("---")

    if api_key:
        tab1, tab2 = st.tabs(["📂 อัปโหลดรูป", "📷 ถ่ายรูป"])

        # TAB 1: Upload
        with tab1:
            uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            if uploaded_files:
                st.success(f"ส่งให้ AI (2.0 Flash) วิเคราะห์ {len(uploaded_files)} รูป...")
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
                                # ถ้า 2.0 ยังติด Error ให้แนะนำ 1.5
                                if "429" in code:
                                    st.caption("โควต้าเต็มชั่วคราว หรือโมเดลนี้ยังไม่เปิดให้ใช้ฟรีเต็มที่ ลองเปลี่ยนเป็น 'gemini-1.5-flash'")
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

