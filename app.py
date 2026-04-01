import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, butter, lfilter

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="GaitPro AI | Clinical Analysis", layout="wide")

# ปรับฟอนต์และสไตล์ตารางให้ดูสะอาดตาและเป็นมืออาชีพขึ้น
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif;
    }
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-box { padding: 10px; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_value=True)

# ฟังก์ชันกรองสัญญาณ Noise
def butter_lowpass_filter(data, cutoff=20, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🩺 GaitPro AI")
    st.markdown("**ระบบวิเคราะห์การเดินเชิงคลินิก**")
    uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
    st.divider()
    patient_weight = st.number_input("น้ำหนักตัว (kg)", value=70.0)
    step_height = st.number_input("ระยะยกตัวแนวดิ่ง (m)", value=0.45, help="ระยะขณะลุกขึ้นยืน [Ref: Prototype Fallsense p.7]")
    st.divider()
    st.info("💡 วิเคราะห์ตามเกณฑ์มาตรฐาน FallSense Clinical Interpretation [Ref: Prototype Fallsense p.8-9]")

# สร้าง Tabs
tab_analysis, tab_manual = st.tabs(["📊 Analysis Dashboard", "📖 User Manual & References"])

# --- 3. TAB: USER MANUAL & INTERPRETATION ---
with tab_manual:
    st.header("📖 คู่มือการใช้งานและเกณฑ์อ้างอิงทางการแพทย์")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("💡 ขั้นตอนการใช้งาน")
        st.info("""
        1. **การติดตั้ง:** ติดเซนเซอร์ที่เอวด้านหลัง (Waist L3-L5) ใกล้จุดศูนย์กลางมวล [Ref: รายละเอียด p.2]
        2. **การบันทึก:** ใช้ Sampling Rate 50-100 Hz [Ref: รายละเอียด p.2]
        3. **Protocol การเดิน:** เดินทางตรง 10-20 วินาที เพื่อวิเคราะห์ Gait [Ref: รายละเอียด p.1]
        4. **Protocol ลุกนั่ง:** ลุก-นั่ง 5 ครั้ง เพื่อประเมิน Sarcopenia [Ref: รายละเอียด p.1]
        """)
        
    with col_m2:
        st.subheader("🧬 เกณฑ์การแปรผลอ้างอิง")
        # ตารางเกณฑ์การแปรผลพร้อมระบุแหล่งอ้างอิง
        ref_grid = {
            "ตัวแปร (Parameter)": ["Gait Speed", "Stride Variability", "RMS Trunk Sway", "STS Power"],
            "เกณฑ์ปกติ": ["> 1.0 m/s", "< 3%", "1.5 - 2.5 m/s²", "> 300 W"],
            "เสี่ยงสูง": ["< 0.8 m/s", "> 5%", "> 2.5 m/s²", "< 200 W"],
            "แหล่งอ้างอิง (Source)": [
                "Prototype Fallsense p.8",
                "Prototype Fallsense p.8",
                "Prototype Fallsense p.8",
                "Prototype Fallsense p.9"
            ]
        }
        st.table(pd.DataFrame(ref_grid))

    st.divider()
    st.subheader("🧠 คำอธิบายตัวแปรวิจัย")
    st.write("- **Stride Variability (CV%):** วัดความเสถียรของ Gait Rhythm [Ref: Prototype Fallsense p.8]")
    st.write("- **RMS Trunk Sway:** วัดการแกว่งของลำตัว (Postural Sway) [Ref: Prototype Fallsense p.8]")
    st.write("- **STS Power:** ประเมินความแข็งแรงกล้ามเนื้อขา (Leg Strength) [Ref: Prototype Fallsense p.9]")

# --- 4. TAB: ANALYSIS DASHBOARD ---
with tab_analysis:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        fs = 100 

        # A. Processing
        df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['mag_filtered'] = butter_lowpass_filter(df['mag'], cutoff=20, fs=fs)

        # B. Step Detection
        peaks, _ = find_peaks(df['mag_filtered'], height=10.5, distance=fs//2)
        stride_times = np.diff(df['timestamp'].iloc[peaks[::2]]) if len(peaks) > 2 else []

        # C. Metrics Calculation
        cv = (np.std(stride_times) / np.mean(stride_times)) * 100 if len(stride_times) > 0 else 0
        rms_sway = np.sqrt(np.mean(df['mag']**2))
        sts_time = 1.2 
        sts_power = (patient_weight * 9.81 * step_height) / sts_time

        st.header("📊 ผลการวิเคราะห์รายบุคคล")
        
        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Stride Variability", f"{cv:.1f}%", delta="Stable" if cv < 3 else "Risk", delta_color="inverse")
        with m2:
            st.metric("RMS Trunk Sway", f"{rms_sway:.2f}", delta="Normal" if rms_sway <= 2.5 else "High", delta_color="inverse")
        with m3:
            st.metric("STS Power", f"{int(sts_power)}W", delta="Good" if sts_power > 300 else "Low")
        with m4:
            risk_score = (cv * 10) + (rms_sway * 5)
            st.metric("Fall Risk Index", f"{int(risk_score)}/100")

        st.divider()

        # ส่วนตารางสรุปผลที่มี References และสีสัน
        st.subheader("📋 ตารางสรุปการประเมินเชิงคลินิก")
        
        # ฟังก์ชันกำหนดสีและข้อความ
        def get_status(val, param):
            if param == "CV":
                return "🔴 เสี่ยงล้มสูง" if val > 5 else "🟡 เริ่มไม่มั่นคง" if val > 3 else "🟢 ปกติ"
            if param == "RMS":
                return "🔴 Balance ไม่ดี" if val > 2.5 else "🟢 ปกติ"
            if param == "Power":
                return "🔴 เสี่ยง Sarcopenia" if val < 200 else "🟢 ปกติ"
            return ""

        summary_df = pd.DataFrame({
            "ตัวแปรวิเคราะห์": ["Stride Variability", "RMS Trunk Sway", "STS Power"],
            "ค่าที่วัดได้": [f"{cv:.1f}%", f"{rms_sway:.2f}", f"{int(sts_power)}W"],
            "สถานะ (Status)": [get_status(cv, "CV"), get_status(rms_sway, "RMS"), get_status(sts_power, "Power")],
            "อ้างอิงเกณฑ์วัด (Reference)": ["Prototype Fallsense p.8", "Prototype Fallsense p.8", "Prototype Fallsense p.9"]
        })
        st.table(summary_df)

        # Visualizations
        st.subheader("📉 กราฟแสดงสัญญาณความเร่ง (Motion Waveform)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag_filtered'], name="Filtered Acceleration", line=dict(color="#007AFF")))
        fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag_filtered'].iloc[peaks], mode='markers', name="Step Point", marker=dict(color="red", size=8)))
        fig.update_layout(xaxis_title="เวลา (s)", yaxis_title="ความเร่ง (m/s²)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👋 ยินดีต้อนรับ! กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการวิเคราะห์ตามเกณฑ์มาตรฐาน [Ref: รายละเอียด p.2]")
