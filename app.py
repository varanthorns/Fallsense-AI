import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, butter, lfilter

# --- 1. SETTINGS & CLINICAL THRESHOLDS ---
st.set_page_config(page_title="GaitPro AI | Clinical Analysis", layout="wide")

# ฟังก์ชันกรองสัญญาณ Noise (Low-pass Filter) ตามเอกสาร [cite: 65, 66]
def butter_lowpass_filter(data, cutoff=20, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🩺 GaitPro AI")
    uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
    st.divider()
    patient_weight = st.number_input("Patient Weight (kg)", value=70.0)
    step_height = st.number_input("Vertical Rise (m)", value=0.45, help="ระยะยกตัวแนวดิ่งขณะลุกยืน [cite: 127]")
    st.divider()
    st.info("ระบบจะวิเคราะห์ตาม Clinical Interpretation ของ FallSense Belt [cite: 156]")

# สร้าง Tabs สำหรับแบ่งหน้าจอ
tab_analysis, tab_manual = st.tabs(["📊 Analysis Dashboard", "📖 User Manual & Interpretation"])

# --- 3. TAB: USER MANUAL & INTERPRETATION (หน้าคู่มือ) ---
with tab_manual:
    st.header("📖 คู่มือการใช้งานและเกณฑ์การแปรผลทางการแพทย์")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("💡 ขั้นตอนการใช้งานอุปกรณ์")
        st.write("""
        1. **การติดตั้ง:** ติดตั้งอุปกรณ์บริเวณเอวด้านหลัง (Waist L3-L5) ซึ่งเป็นจุดศูนย์กลางมวล [cite: 51]
        2. **การบันทึก:** บันทึกข้อมูลด้วย Sampling Rate 50-100 Hz [cite: 6, 63]
        3. **กิจกรรมที่แนะนำ:** * เดินทางตรง 10-20 วินาที (วิเคราะห์ Gait) [cite: 4]
            * ลุก-นั่ง 5 ครั้ง (วิเคราะห์ Power/Sarcopenia) [cite: 4, 115]
        4. **การนำเข้า:** ดาวน์โหลดไฟล์เป็น .csv แล้วนำมาอัปโหลดในระบบนี้ [cite: 6]
        """)
        
    with col_m2:
        st.subheader("🧬 เกณฑ์การแปรผล (Clinical Interpretation)")
        # ตารางเกณฑ์การแปรผลตามเอกสาร [cite: 154, 159, 162, 165, 174]
        interpretation_grid = {
            "ตัวแปร (Parameter)": ["Gait Speed", "Stride Variability", "RMS Trunk Sway", "STS Power"],
            "ปกติ (Normal)": ["> 1.0 m/s", "< 3%", "1.5 - 2.5 m/s²", "> 300 W"],
            "เริ่มผิดปกติ (Borderline)": ["0.8 - 1.0 m/s", "3 - 5%", "-", "200 - 300 W"],
            "ความเสี่ยงสูง (High Risk)": ["< 0.8 m/s", "> 5%", "> 2.5 m/s²", "< 200 W"]
        }
        st.table(pd.DataFrame(interpretation_grid))

    st.divider()
    st.subheader("🧠 คำอธิบายตัวแปร")
    st.markdown("""
    * **Stride Variability (CV%):** วัดความสม่ำเสมอของการเดิน ค่าที่สูงบ่งชี้ถึงความไม่มั่นคงและเสี่ยงต่อการล้ม [cite: 99, 160]
    * **RMS Trunk Sway:** วัดการแกว่งของลำตัว ค่าที่สูงสัมพันธ์กับความเสี่ยงหกล้ม [cite: 113, 163]
    * **STS Power:** กำลังกล้ามเนื้อขา ค่าที่ต่ำบ่งชี้ภาวะกล้ามเนื้อพร่อง (Sarcopenia) [cite: 129, 172]
    """)

# --- 4. TAB: ANALYSIS DASHBOARD (หน้าวิเคราะห์ผล) ---
with tab_analysis:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        fs = 100 # [cite: 63]

        # A. ประมวลผลสัญญาณ [cite: 73, 74]
        df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['mag_filtered'] = butter_lowpass_filter(df['mag'], cutoff=20, fs=fs)

        # B. ตรวจจับก้าวเดิน [cite: 76, 83, 85]
        peaks, _ = find_peaks(df['mag_filtered'], height=10.5, distance=fs//2)
        stride_times = np.diff(df['timestamp'].iloc[peaks[::2]]) if len(peaks) > 2 else []

        # C. คำนวณตัวแปรสำคัญ [cite: 98, 108, 123]
        cv = (np.std(stride_times) / np.mean(stride_times)) * 100 if len(stride_times) > 0 else 0
        rms_sway = np.sqrt(np.mean(df['mag']**2))
        sts_time = 1.2 # จำลอง
        sts_power = (patient_weight * 9.81 * step_height) / sts_time

        st.header("📊 Clinical Gait Analysis Dashboard")
        
        # ส่วน Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            status = "✅ Stable" if cv < 3 else "🟡 Borderline" if cv <= 5 else "🔴 Unstable"
            st.metric("Stride Variability", f"{cv:.1f}%", delta=status, delta_color="inverse")
        with m2:
            sway_status = "✅ Normal" if rms_sway <= 2.5 else "🔴 Impairment"
            st.metric("RMS Trunk Sway", f"{rms_sway:.2f}", delta=sway_status, delta_color="inverse")
        with m3:
            power_status = "✅ Strong" if sts_power > 300 else "🔴 Sarcopenia Risk" if sts_power < 200 else "🟡 Normal"
            st.metric("STS Power", f"{int(sts_power)}W", delta=power_status)
        with m4:
            risk_score = (cv * 10) + (rms_sway * 5)
            risk_level = "High" if risk_score > 60 else "Moderate" if risk_score > 30 else "Low"
            st.metric("Fall Risk Score", f"{int(risk_score)}/100", delta=risk_level, delta_color="inverse")

        st.divider()

        # ส่วนตารางสรุปผล
        st.subheader("📋 ผลการประเมินรายบุคคล")
        analysis_data = {
            "ตัวแปร": ["Stride Variability", "RMS Trunk Sway", "STS Power"],
            "ค่าที่วัดได้": [f"{cv:.1f}%", f"{rms_sway:.2f}", f"{int(sts_power)}W"],
            "ผลการประเมิน": [
                "🔴 เสี่ยงล้มสูง (Instability)" if cv > 5 else "✅ ปกติ",
                "🔴 Balance Impairment" if rms_sway > 2.5 else "✅ ปกติ",
                "🔴 Sarcopenia Risk" if sts_power < 200 else "✅ ปกติ"
            ]
        }
        st.table(pd.DataFrame(analysis_data))

        # ส่วนกราฟ
        st.subheader("📉 Motion Data Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag_filtered'], name="Filtered Accel", line=dict(color="#007AFF")))
        fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag_filtered'].iloc[peaks], mode='markers', name="Step Detected", marker=dict(color="red", size=8)))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Magnitude (m/s²)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👋 ยินดีต้อนรับสู่ GaitPro AI! กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการวิเคราะห์ หรือศึกษาคู่มือที่ Tab ด้านบน [cite: 186]")
