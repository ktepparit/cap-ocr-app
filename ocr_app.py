import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Kratingdaeng AI Scanner", page_icon="⚡", layout="centered")

# --- ส่วนใส่ API Key ---
with st.sidebar:
    st.header("🔑 ตั้งค่าระบบ")
    st.info("ใช้สมอง AI ของ Google Gemini (Pure Vision)")
    api_key = st.text_input("ใส่ Google API Key", type="password")
    
    if not api_key:
        st.warning("⚠️ ต้องใส่ API Key ก่อนใช้งาน")
        st.markdown("[👉 กดขอ API Key ฟรีที่นี่](https://aistudio.google.com/app/apikey)")

# --- ฟังก์ชันอ่านภาพด้วย Gemini (แบบ Clean Prompt) ---
def gemini_vision_scan(image_pil, key):
    try:
        # ตั้งค่าโมเดล
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # --- Clean Prompt (คำสั่งแบบกลางๆ ไม่ชี้นำ) ---
        # เราบอกแค่ "โครงสร้างข้อมูล" (12 หลัก) และ "สิ่งที่ต้องตัดทิ้ง" (ขยะ)
        # แต่ไม่บอกว่าต้องแปลงตัวอักษรไหนเป็นตัวไหน ให้ AI ตัดสินใจเองจากภาพ
        prompt = """
        Look at this image of a bottle cap.
        There is a code printed on the inside surface.
        
        Please extract the code following these criteria:
        1. The code contains exactly 12 alphanumeric characters (A-Z and 0-9).
        2. Ignore unrelated text such as "P Bev", "21", "HDPE", plastic recycling symbols, or numbers denoting cap size.
        3. Focus only on the main 12-character serial code.
        
        Output ONLY the text of the code. Do not add any explanation.
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
    st.caption("Mode: Pure AI (No Correction Rules)")
    st.write("---")

    if api_key:
        tab1, tab2 = st.tabs(["📂 อัปโหลดรูป", "📷 ถ่ายรูป"])

        # TAB 1: Upload
        with tab1:
            uploaded_files = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            if uploaded_files:
                st.success(f"กำลังส่ง {len(uploaded_files)} รูปไปให้ AI ดู...")
                st.markdown("---")
                for i, uploaded_file in enumerate(uploaded_files):
                    col1, col2 = st.columns([1, 3])
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, width=100, caption=f"Img {i+1}")
                    with col2:
                        with st.spinner('AI กำลังแกะรอย...'):
                            code = gemini_vision_scan(image, api_key)
                            
                            if "Error" in code:
                                st.error(code)
                            else:
                                # แสดงผลลัพธ์ดิบๆ จาก AI
                                st.code(code, language=None)
                                
                                # ตรวจสอบความยาวแค่เบื้องต้น
                                clean_code = code.replace(" ", "")
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
                with st.spinner('AI กำลังแกะรอย...'):
                    code = gemini_vision_scan(image, api_key)
                    if "Error" in code:
                        st.error(code)
                    else:
                        st.code(code, language=None)
                        clean_code = code.replace(" ", "")
                        if len(clean_code) == 12:
                            st.caption("✅ ครบ 12 หลัก")
                        else:
                            st.caption(f"⚠️ อ่านได้ {len(clean_code)} หลัก")
    else:
        st.info("👈 กรุณาใส่ API Key ทางด้านซ้ายเพื่อเริ่มใช้งาน")

except Exception as main_e:
    st.error(f"Critical: {main_e}")
