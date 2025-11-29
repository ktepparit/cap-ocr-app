import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng AI Scanner", page_icon="⚡", layout="centered")

# --- ส่วนใส่ API Key ---
with st.sidebar:
    st.header("🔑 ตั้งค่าระบบ")
    st.success("Model: gemini-pro-latest") # ใช้ตัว Pro รุ่นล่าสุด
    
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
        
        # ✅ ใช้โมเดล gemini-pro-latest ตามที่ขอ
        model = genai.GenerativeModel('gemini-pro-latest')

        # --- Prompt แบบละเอียดสำหรับรุ่น Pro (สั่งให้วิเคราะห์บริบท) ---
        prompt = """
        You are an advanced AI reading a serial code on a bottle cap.
        The text is in a DOT-MATRIX font, which often causes specific OCR errors.
        
        YOUR TASK: Extract the exactly 12-character alphanumeric code.

        CORRECTION RULES (Apply these logic steps):
        1. **'7' vs 'Z':** In this specific font, the number '7' has a curved top that looks like 'Z'. Given the context of these codes, it is almost ALWAYS '7', not 'Z'.
        2. **'6' vs 'G':** The number '6' often has a gap at the top loop, resembling 'G'. Check closely.
        3. **'W' vs 'I' or 'U':** The letter 'W' is wide and formed by faint dots. Do not mistake it for a narrow 'I'.
        4. **'M' vs 'H':** Look for the central dip of 'M'.
        
        OUTPUT FORMAT:
        - The code must be exactly 12 characters.
        - Characters allowed: A-Z, 0-9.
        - Ignore: "P Bev", "21", "HDPE", recycling symbols.
        - Output ONLY the 12-character code string.
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
    st.caption("Powered by: Gemini Pro Latest 🧠") 
    st.write("---")

    if api_key:
        tab1, tab2 = st.tabs(["📂 อัปโหลดรูป", "📷 ถ่ายรูป"])

        # TAB 1: Upload
        with tab1:
            uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            if uploaded_files:
                st.success(f"ส่งให้ AI (Pro Latest) วิเคราะห์ {len(uploaded_files)} รูป...")
                st.markdown("---")
                for i, uploaded_file in enumerate(uploaded_files):
                    col1, col2 = st.columns([1, 3])
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, width=100, caption=f"Img {i+1}")
                    with col2:
                        with st.spinner('AI กำลังใช้ความคิด...'):
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
