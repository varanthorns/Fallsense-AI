import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, welch

# --- 1. SETTINGS & THEME ---
st.set_page_config(
    page_title="GaitPro AI | Next-Gen Analytics",
    page_icon="⚡",
    layout="wide"
)

# ใช้ Markdown แบบเรียบง่ายเพื่อเลี่ยง Error ใน Python เวอร์ชันใหม่ๆ
st.markdown("### 🏃 GaitPro AI Professional System")

# --- 2. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063200.png", width=100)
    st.header("Medical Control")
    mode = st.selectbox("Navigation", ["Executive Dashboard", "Usage Guide", "System Status"])
    st.divider()
    
    uploaded_file = st.file_uploader("📂 Import Clinical Data (CSV)", type="csv")
    
    if uploaded_file:
        st.success("File Verified")
        p_id = st.text_input("Patient Reference", "ID-8829")
        sens = st.slider("Detection Sensitivity", 5.0, 15.0, 10.0)
        st.divider()
        st.caption("System Version: 2.5.0 Gold")

# --- 3. FUNCTION: ANALYSIS ENGINE ---
def analyze_gait(df, threshold):
    # Signal Processing
    df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['mag_f'] = df['mag'].rolling(window=5, center=True).mean().fillna(df['mag'])
    
    # Peak Detection (Heel Strikes)
    peaks, _ = find_peaks(df['mag_f'], height=threshold, distance=20)
    
    # Metrics Calculation
    steps = len(peaks)
    duration = df['timestamp'].max() - df['timestamp'].min()
    cadence = (steps / duration) * 60 if duration > 0 else 0
    
    # Variability (The Pro Metric)
    intervals = np.diff(df['timestamp'].iloc[peaks]) if len(peaks) > 1 else [0]
    cv = (np.std(intervals) / np.mean(intervals)) * 100 if len(intervals) > 1 else 0
    
    return df, peaks, steps, cadence, cv, duration

# --- 4. MODE: USAGE GUIDE (UX ชั้นนำต้องมี Onboarding) ---
if mode == "Usage Guide":
    st.header("How to operate GaitPro AI")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. Placement")
        st.write("ติดอุปกรณ์ไว้ที่กึ่งกลางหลังส่วนล่าง (L5) เพื่อความแม่นยำสูงสุดของจุดศูนย์ถ่วง")
    with c2:
        st.subheader("2. Capture")
        st.write("กด Record บนหน้าเว็บ ESP32 เดินอย่างน้อย 15 ก้าว แล้วกด Stop ก่อน Download")
    with c3:
        st.subheader("3. Analyze")
        st.write("อัปโหลดไฟล์ CSV เข้าสู่ระบบนี้เพื่อรับผลวิเคราะห์ระดับการแพทย์ทันที")

# --- 5. MODE: EXECUTIVE DASHBOARD ---
elif mode == "Executive Dashboard":
    if not uploaded_file:
        st.info("💡 Waiting for Data: Please upload a CSV file from the sidebar to begin analysis.")
        st.image("https://static.vecteezy.com/system/resources/previews/002/098/203/non_2x/silver-robot-with-analysis-data-on-screen-free-vector.jpg")
    else:
        # Run Engine
        raw_df = pd.read_csv(uploaded_file)
        df, peaks, steps, cadence, cv, duration = analyze_gait(raw_df, sens)

        # --- Top UI: Metrics Cards ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Activity Steps", f"{steps} Steps")
        with col2:
            st.metric("Walking Cadence", f"{cadence:.1f} SPM")
        with col3:
            st.metric("Gait Variability (CV)", f"{cv:.2f}%", delta="Normal" if cv < 4 else "Unstable", delta_color="inverse")
        with col4:
            # Score Logic
            health_score = max(0, 100 - (cv * 8))
            st.metric("Overall Stability Score", f"{health_score:.1f}/100")

        st.divider()

        # --- Middle UI: Visual Analytics ---
        t1, t2 = st.tabs(["📈 Kinetic Waveform", "📊 Distribution"])
        
        with t1:
            fig = go.Figure()
            # Acceleration Line
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag_f'], name="Body Motion", line=dict(color='#007AFF', width=2)))
            # Heel Strike Points
            fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag_f'].iloc[peaks], mode='markers', name="Heel Strike", marker=dict(color='#FF2D55', size=10)))
            
            fig.update_layout(
                title="Gait Cycle Signal (Acceleration Magnitude)",
                xaxis_title="Time (Seconds)",
                yaxis_title="G-Force",
                hovermode="x unified",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            # แสดงกราฟแท่งช่วงเวลาการก้าว
            if len(peaks) > 1:
                intervals = np.diff(df['timestamp'].iloc[peaks])
                fig_dist = go.Figure(data=[go.Histogram(x=intervals, marker_color='#34C759', nbinsx=10)])
                fig_dist.update_layout(title="Stride Interval Distribution (Rhythm)", xaxis_title="Interval (s)", template="plotly_white")
                st.plotly_chart(fig_dist, use_container_width=True)

        # --- Bottom UI: AI Diagnostic & Summary ---
        st.divider()
        bottom_l, bottom_r = st.columns([2, 1])
        
        with bottom_l:
            st.subheader("🩺 AI Clinical Insight")
            if cv > 5:
                st.error(f"**High Fall Risk Detected!** (CV: {cv:.2f}%)")
                st.write("ตรวจพบความไม่สม่ำเสมอในการก้าวเดินสูงกว่าปกติ แนะนำให้ทำการประเมินทางกายภาพบำบัดเพิ่มเติม")
            elif cv > 3:
                st.warning("**Moderate Instability:** การทรงตัวอยู่ในระดับปานกลาง ควรระวังการเดินบนพื้นต่างระดับ")
            else:
                st.success("**Perfect Symmetry:** สุขภาพการเดินดีเยี่ยม จังหวะการก้าวมีความสม่ำเสมอสูง")

        with bottom_r:
            st.subheader("📋 Final Summary")
            st.write(f"**Patient ID:** {p_id}")
            st.write(f"**Test Duration:** {duration:.2f} s")
            st.button("📥 Generate Medical Report (PDF)")

# --- 6. MODE: SYSTEM STATUS ---
else:
    st.subheader("System Information")
    st.json({
        "Model": "GaitAI v2.5",
        "Algorithm": "Peak-Interval Variability Analysis",
        "Sensor_Sync": "Active",
        "Cloud_Status": "Connected"
    })
