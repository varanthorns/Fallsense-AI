import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, butter, lfilter

# --- 1. SETTINGS ---
st.set_page_config(page_title="GaitPro AI | Clinical Analysis", layout="wide")

# ฟังก์ชันกรองสัญญาณ Noise
def butter_lowpass_filter(data, cutoff=20, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🩺 GaitPro AI")
    st.write("---")
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ข้อมูล (CSV)", type="csv")
    st.write("---")
    patient_weight = st.number_input("น้ำหนักตัว (kg)", value=70.0)
    step_height = st.number_input("ระยะยกตัวแนวดิ่ง (m)", value=0.45)


# สร้าง Tabs สำหรับการใช้งาน
tab_analysis, tab_manual = st.tabs(["📊 Analysis Dashboard", "📖 User Manual & References"])

# --- 3. TAB: USER MANUAL & REFERENCES ---
with tab_manual:
    st.header("📖 คู่มือและการตีความผลเชิงคลินิก")
    
    # ส่วนวิธีใช้
    with st.expander("✅ ขั้นตอนการใช้งานอุปกรณ์ และ Protocol", expanded=True):
        st.markdown("""
        1. **การติดตั้ง:** ติดเซนเซอร์ที่เอวด้านหลัง (L3-L5) เพื่อให้อยู่ใกล้ Center of Mass 
        2. **การบันทึก:** ตั้งค่า Sampling Rate ที่ 50-100 Hz 
        3. **Walking Protocol:** เดินทางตรงปกติ 10-20 วินาที 
        4. **STS Protocol:** ลุก-นั่งจากเก้าอี้ 5 ครั้งต่อเนื่อง
        """)

    # ส่วนตารางเกณฑ์วัด (เพิ่มคอลัมน์อ้างอิงชัดเจน)
    st.subheader("🧬 Clinical Reference Grid")
    ref_data = {
        "ตัวแปร (Parameter)": ["Gait Speed", "Stride Variability", "RMS Trunk Sway", "STS Power"],
        "เกณฑ์ปกติ (Normal)": ["> 1.0 m/s", "< 3%", "1.5 - 2.5", "> 300 W"],
        "ความเสี่ยงสูง (High Risk)": ["< 0.8 m/s", "> 5%", "> 2.5", "< 200 W"],
        "แหล่งอ้างอิง (Source)": [
            "Prototype Fallsense p.8",
            "Prototype Fallsense p.8",
            "Prototype Fallsense p.8",
            "Prototype Fallsense p.9"
        ]
    }
    st.table(pd.DataFrame(ref_data))

    st.markdown("""
    **คำอธิบายเพิ่มเติม:**
    * **Stride Variability (CV%):** วัดความสม่ำเสมอของจังหวะก้าว 
    * **RMS Trunk Sway:** วัดความเสถียรของลำตัวขณะเคลื่อนที่ 
    * **STS Power:** วัดกำลังกล้ามเนื้อขาเพื่อประเมินภาวะ Sarcopenia 
    """)

# --- 4. TAB: ANALYSIS DASHBOARD ---
with tab_analysis:
    if not uploaded_file:
        st.warning("⚠️ กรุณาอัปโหลดไฟล์ CSV ที่ Sidebar ด้านซ้ายเพื่อเริ่มการวิเคราะห์")
        st.stop()

    # --- PROCESSING ---
    df = pd.read_csv(uploaded_file)
    # --- เพิ่มส่วนนี้เพื่อล้างชื่อคอลัมน์ (Clean Column Names) ---
    df.columns = df.columns.str.strip() # ลบช่องว่างหน้าและหลังชื่อ (เช่น ' ax ' -> 'ax')
    df.columns = df.columns.str.lower() # แปลงเป็นตัวพิมพ์เล็กทั้งหมด (เช่น 'AX' -> 'ax')
    fs = 100 
    # -------------------------------------------------------
        
        # ตอนนี้บรรทัดนี้จะทำงานได้ปกติ ไม่ติด KeyError แล้วครับ
        df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    fs = 100 
    
    # คำนวณ Magnitude และ Filter
    df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['mag_f'] = butter_lowpass_filter(df['mag'], cutoff=20, fs=fs)

    # ตรวจจับก้าว
    peaks, _ = find_peaks(df['mag_f'], height=10.5, distance=fs//2)
    stride_times = np.diff(df['timestamp'].iloc[peaks[::2]]) if len(peaks) > 2 else []

    # คำนวณค่าทางคลินิก
    cv = (np.std(stride_times) / np.mean(stride_times)) * 100 if len(stride_times) > 0 else 0
    rms_sway = np.sqrt(np.mean(df['mag']**2))
    # จำลองค่า STS Time (ในงานจริงควรดึงจากช่วงกิจกรรมลุกนั่ง)
    sts_power = (patient_weight * 9.81 * step_height) / 1.2 

    # --- DISPLAY ---
    st.header("📊 Clinical Analysis Result")

    # แถบ Metrics ด้านบน
    col1, col2, col3 = st.columns(3)
    col1.metric("Stride Variability", f"{cv:.1f}%", help="ค่าปกติควร < 3%")
    col2.metric("RMS Trunk Sway", f"{rms_sway:.2f}", help="ค่าปกติคือ 1.5 - 2.5")
    col3.metric("STS Power", f"{int(sts_power)}W", help="ค่าต่ำกว่า 200W เสี่ยง Sarcopenia")

    st.write("---")

    # ตารางสรุปผลที่มี References และแยกสถานะ
    st.subheader("📋 ตารางประเมินผลเชิงคลินิก")
    
    def format_status(val, p_type):
        if p_type == "CV":
            return "🔴 เสี่ยงล้มสูง" if val > 5 else "🟡 เริ่มผิดปกติ" if val > 3 else "🟢 ปกติ"
        if p_type == "RMS":
            return "🔴 Balance ไม่ดี" if val > 2.5 else "🟢 ปกติ"
        if p_type == "Power":
            return "🔴 เสี่ยง Sarcopenia" if val < 200 else "🟢 ปกติ"
        return ""

    final_results = pd.DataFrame({
        "ตัวแปรวิเคราะห์": ["Stride Variability (CV%)", "RMS Trunk Sway", "Sit-to-Stand Power"],
        "ค่าที่วัดได้": [f"{cv:.1f} %", f"{rms_sway:.2f}", f"{int(sts_power)} W"],
        "สถานะการประเมิน": [format_status(cv, "CV"), format_status(rms_sway, "RMS"), format_status(sts_power, "Power")],
        "อ้างอิงเกณฑ์ (Reference)": ["Prototype Fallsense p.8", "Prototype Fallsense p.8", "Prototype Fallsense p.9"]
    })
    
    st.table(final_results)

    # กราฟ
    st.subheader("📉 กราฟสัญญาณความเร่ง (Motion Signal)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag_f'], name="ความเร่งรวม (Filtered)", line=dict(color="#007AFF")))
    fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag_f'].iloc[peaks], mode='markers', name="จุดตรวจพบก้าว", marker=dict(color="red", size=8)))
    fig.update_layout(xaxis_title="เวลา (วินาที)", yaxis_title="ความเร่ง (m/s²)", template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)
