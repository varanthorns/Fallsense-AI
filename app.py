import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, butter, lfilter

# --- 1. SETTINGS ---
st.set_page_config(page_title="GaitPro AI | Clinical Analysis", layout="wide")

# ฟังก์ชันกรองสัญญาณ Noise [Ref: รายละเอียด.pdf p.2]
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
    st.info("ระบบอ้างอิงเกณฑ์ FallSense Prototype")

# สร้าง Tabs
tab_analysis, tab_manual = st.tabs(["📊 Analysis Dashboard", "📖 User Manual & References"])

# --- 3. TAB: USER MANUAL & REFERENCES ---
with tab_manual:
    st.header("📖 คู่มือและการตีความผลเชิงคลินิก")
    
    col_ref1, col_ref2 = st.columns(2)
    with col_ref1:
        st.subheader("✅ ขั้นตอนการทดสอบ")
        st.markdown("""
        1. **ติดตั้งเซนเซอร์:** ที่ระดับเอว L3-L5 (Center of Mass)
        2. **เดิน (Gait):** เดินทางตรงปกติ 10-20 วินาที
        3. **ลุก-นั่ง (STS):** ทำต่อเนื่อง 5 ครั้ง [Ref: p.9]
        """)
    
    with col_ref2:
        st.subheader("🧬 เกณฑ์อ้างอิง (Reference)")
        ref_data = {
            "ตัวแปร": ["Stride Var", "Trunk Sway", "STS Power"],
            "เกณฑ์ปกติ": ["< 3%", "1.5 - 2.5", "> 300 W"],
            "หน้าอ้างอิง": ["P.8", "P.8", "P.9"]
        }
        st.table(pd.DataFrame(ref_data))

# --- 4. TAB: ANALYSIS DASHBOARD ---
with tab_analysis:
    if not uploaded_file:
        st.info("👋 กรุณาอัปโหลดไฟล์ CSV ที่แถบด้านซ้ายเพื่อเริ่มการวิเคราะห์")
        st.stop()

    # --- PROCESSING ---
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        # ล้างชื่อคอลัมน์ให้สะอาด (ลบช่องว่าง + ตัวพิมพ์เล็ก)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # ตรวจเช็คคอลัมน์สำคัญ
        required = ['timestamp', 'ax', 'ay', 'az']
        if not all(col in df.columns for col in required):
            st.error(f"❌ คอลัมน์ไม่ครบ! ไฟล์ต้องมี: {', '.join(required)}")
            st.stop()

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
        sts_power = (patient_weight * 9.81 * step_height) / 1.2 

        # --- DISPLAY ---
        st.header("📊 Clinical Analysis Result")

        # แสดง Metrics (แบบดั้งเดิมของ Streamlit ที่ไม่พังแน่นอน)
        m1, m2, m3 = st.columns(3)
        m1.metric("Stride Variability", f"{cv:.1f}%")
        m2.metric("RMS Trunk Sway", f"{rms_sway:.2f}")
        m3.metric("STS Power", f"{int(sts_power)}W")

        st.divider()

        # ตารางสรุปสถานะ
        st.subheader("📋 บทสรุปการประเมิน")
        
        def get_status(val, type):
            if type == "CV": return "🟢 ปกติ" if val < 3 else "🔴 เสี่ยงล้ม"
            if type == "RMS": return "🟢 ปกติ" if 1.5 <= val <= 2.5 else "🔴 ทรงตัวไม่นิ่ง"
            if type == "Power": return "🟢 แข็งแรง" if val > 300 else "🔴 กล้ามเนื้อพร่อง"
            return ""

        res_table = pd.DataFrame({
            "ตัวแปร": ["Stride Variability (CV%)", "RMS Trunk Sway", "Sit-to-Stand Power"],
            "ค่าที่ได้": [f"{cv:.1f}%", f"{rms_sway:.2f}", f"{int(sts_power)}W"],
            "สถานะ": [get_status(cv, "CV"), get_status(rms_sway, "RMS"), get_status(sts_power, "Power")],
            "อ้างอิง": ["P.8", "P.8", "P.9"]
        })
        st.table(res_table)

        # กราฟ
        st.subheader("📉 Motion Waveform")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag_f'], name="ความเร่ง (Filtered)", line=dict(color="#007AFF")))
        fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag_f'].iloc[peaks], mode='markers', name="จุดตรวจพบก้าว", marker=dict(color="red", size=8)))
        fig.update_layout(xaxis_title="เวลา (วินาที)", yaxis_title="ความเร่ง (m/s²)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
