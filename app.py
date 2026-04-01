import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

# --- Config หน้าเว็บ ---
st.set_page_config(page_title="GaitAI - Medical Dashboard", layout="wide")

st.title("🏃 GaitAI: Clinical Fall Risk & Mobility Analysis")
st.markdown("---")

# --- Sidebar: อัปโหลดและตั้งค่า ---
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV from ESP32", type="csv")

st.sidebar.header("👤 Patient Profile")
weight = st.sidebar.number_input("Weight (kg)", value=70)
height = st.sidebar.number_input("Height (m)", value=1.70)

if uploaded_file:
    # 1. โหลดข้อมูล
    df = pd.read_csv(uploaded_file)
    
    # 2. คำนวณเบื้องต้น (Signal Processing)
    # หาค่า Magnitude ของความเร่ง (Vector Sum)
    df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    
    # หา Peaks (ก้าวเท้า) - ปรับ height ตามความเหมาะสมของข้อมูลจริง
    peaks, _ = find_peaks(df['mag'], height=11, distance=20) 
    step_count = len(peaks)
    
    # คำนวณ Stride Variability (CV)
    if step_count > 2:
        intervals = np.diff(df['timestamp'].iloc[peaks])
        stride_cv = (np.std(intervals) / np.mean(intervals)) * 100
    else:
        stride_cv = 0

    # 3. ส่วนแสดงผล Metrics ด้านบน
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", f"{step_count} steps")
    with col2:
        cadence = round(step_count / (df['timestamp'].max() / 60), 1)
        st.metric("Cadence", f"{cadence} steps/min")
    with col3:
        rms = round(np.sqrt(np.mean(df['mag']**2)), 2)
        st.metric("RMS Acceleration", f"{rms} m/s²")
    with col4:
        # Fall Risk Logic (Simplified AI)
        risk_score = (stride_cv * 0.5) + (rms * 0.2) # สูตรสมมติเพื่อโชว์ Logic
        risk_score = min(max(risk_score/10, 0.1), 1.0) # Normalize 0-1
        st.metric("Fall Risk Score", f"{risk_score:.2f}")

    # 4. ส่วนกราฟ Interactive (Plotly)
    st.subheader("📊 Gait Cycle Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag'], name="Magnitude", line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag'].iloc[peaks], 
                             mode='markers', name="Steps Detected", marker=dict(color='red', size=8)))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 5. ส่วน AI Insights & Diagnosis
    st.markdown("---")
    left_info, right_info = st.columns(2)
    
    with left_info:
        st.subheader("🧠 AI Diagnostic Insight")
        if risk_score > 0.7:
            st.error("⚠️ HIGH RISK: พบความแปรปรวนของการก้าวเท้าสูง (High Stride Variability)")
        elif risk_score > 0.3:
            st.warning("🟡 MODERATE RISK: การทรงตัวมีความไม่นิ่งเล็กน้อย")
        else:
            st.success("✅ LOW RISK: สุขภาพการเดินปกติ")

    with right_info:
        st.subheader("💪 Sarcopenia Estimation (Power)")
        # คำนวณ Power เบื้องต้นจากการลุกนั่ง (ถ้ามี Data ส่วนนั้น)
        max_power = round(weight * 9.81 * (height * 0.25), 2) # สูตรสมมติ
        st.write(f"Estimated Lower Limb Power: **{max_power} Watts**")
        st.progress(min(max_power/500, 1.0))

else:
    st.info("👋 ยินดีต้อนรับ! กรุณาอัปโหลดไฟล์ CSV ที่ดาวน์โหลดจากกล่องเพื่อเริ่มการวิเคราะห์")
    # ตัวอย่างตารางที่ต้องการ
    st.write("ตัวอย่าง Format ไฟล์ CSV:")
    st.code("timestamp,ax,ay,az,gx,gy,gz\n0.00,0.1,0.2,9.8,0.01,0.01,0.01")
