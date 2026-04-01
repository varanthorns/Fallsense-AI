import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, butter, lfilter

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="GaitPro AI | Clinical Analysis", layout="wide")

# ปรับปรุง UI ให้ดูสะอาดตา (Native Streamlit)
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #f0f2f6; }
    </style>
    """, unsafe_allow_value=True)

# ฟังก์ชันกรองสัญญาณ Noise (Low-pass Filter) [Ref: รายละเอียด.pdf p.2]
def butter_lowpass_filter(data, cutoff=20, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🩺 GaitPro AI")
    st.subheader("Clinical Gait Analysis")
    st.write("---")
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ข้อมูล (CSV)", type="csv", help="ไฟล์ต้องมีคอลัมน์ timestamp, ax, ay, az")
    st.write("---")
    patient_weight = st.number_input("น้ำหนักตัวผู้ป่วย (kg)", value=70.0, step=0.1)
    step_height = st.number_input("ระยะยกตัวแนวดิ่ง (m)", value=0.45, help="ระยะจากพื้นถึงเอวขณะยืน ลบด้วยขณะนั่ง")
    st.info("💡 ระบบอ้างอิงเกณฑ์มาตรฐานตาม FallSense Prototype (L3-L5 Placement)")

# สร้าง Tabs
tab_analysis, tab_manual = st.tabs(["📊 Analysis Dashboard", "📖 User Manual & References"])

# --- 3. TAB: USER MANUAL & REFERENCES ---
with tab_manual:
    st.header("📖 คู่มือการใช้งานและเกณฑ์การแปรผล")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("💡 ขั้นตอนการใช้งาน")
        st.markdown("""
        1. **การติดตั้ง:** ติดอุปกรณ์ที่เอวด้านหลังระดับกระดูกสันหลัง **L3-L5** [Ref: p.2]
        2. **การตั้งค่า:** ใช้ Sampling Rate **50-100 Hz** [Ref: p.2]
        3. **Walking Protocol:** เดินทางตรงปกติ 10-20 วินาที เพื่อวัด Gait Rhythm
        4. **STS Protocol:** ลุก-นั่ง 5 ครั้ง เพื่อวัดกำลังกล้ามเนื้อขา [Ref: p.9]
        """)
        
    with col_m2:
        st.subheader("🧬 ตารางเกณฑ์อ้างอิงเชิงคลินิก")
        ref_grid = {
            "ตัวแปร (Parameter)": ["Gait Speed", "Stride Variability", "RMS Trunk Sway", "STS Power"],
            "ปกติ (Normal)": ["> 1.0 m/s", "< 3%", "1.5 - 2.5", "> 300 W"],
            "เสี่ยงสูง (High Risk)": ["< 0.8 m/s", "> 5%", "> 2.5", "< 200 W"],
            "อ้างอิง (Source)": ["Prototype p.8", "Prototype p.8", "Prototype p.8", "Prototype p.9"]
        }
        st.table(pd.DataFrame(ref_grid))

    st.divider()
    st.subheader("🧠 คำอธิบายตัวแปรวิจัย")
    st.write("- **Stride Variability (CV%):** วัดความคงที่ของจังหวะก้าว เดินสม่ำเสมอไหม")
    st.write("- **RMS Trunk Sway:** วัดความนิ่งของลำตัว ยิ่งค่ายิ่งสูงยิ่งเสี่ยงล้ม")
    st.write("- **STS Power:** วัดความแข็งแรงของกล้ามเนื้อขา (สำคัญมากในผู้สูงอายุ)")

# --- 4. TAB: ANALYSIS DASHBOARD ---
with tab_analysis:
    if not uploaded_file:
        st.info("👋 ยินดีต้อนรับสู่ GaitPro AI! กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการวิเคราะห์")
        st.image("https://img.icons8.com/clouds/500/medical-doctor.png", width=200)
        st.stop()

    # --- A. DATA LOADING & CLEANING ---
    try:
        # ใช้ utf-8-sig เพื่อป้องกัน Error จาก BOM อักขระพิเศษหัวไฟล์
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        
        # ล้างชื่อคอลัมน์ให้สะอาด (ลบช่องว่าง + แปลงเป็นตัวพิมพ์เล็ก)
        df.columns = df.columns.str.strip().str.lower()
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_cols = ['timestamp', 'ax', 'ay', 'az']
        if not all(c in df.columns for c in required_cols):
            st.error(f"❌ ไฟล์ CSV ขาดคอลัมน์ที่จำเป็น! พบเพียง: {list(df.columns)}")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ ไม่สามารถอ่านไฟล์ได้: {e}")
        st.stop()

    # --- B. PROCESSING ENGINE ---
    fs = 100 # Sampling Rate มาตรฐาน

    # 1. ประมวลผลสัญญาณ (Preprocessing)
    df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['mag_filtered'] = butter_lowpass_filter(df['mag'], cutoff=20, fs=fs)

    # 2. ตรวจจับก้าว (Peak Detection)
    peaks, _ = find_peaks(df['mag_filtered'], height=10.5, distance=fs//2)
    
    # 3. คำนวณค่าทางคลินิก (Metrics Calculation)
    # Stride Variability (CV%)
    stride_times = np.diff(df['timestamp'].iloc[peaks[::2]]) if len(peaks) > 2 else []
    cv = (np.std(stride_times) / np.mean(stride_times)) * 100 if len(stride_times) > 0 else 0
    
    # RMS Trunk Sway (Sway Analysis)
    rms_sway = np.sqrt(np.mean(df['mag']**2))
    
    # Sit-to-Stand Power (Power Analysis)
    # สูตร: P = (m * g * h) / t [Ref: Prototype p.9]
    sts_time = 1.2 # เวลาเฉลี่ยในการลุกยืน
    sts_power = (patient_weight * 9.81 * step_height) / sts_time

    # --- C. DISPLAY DASHBOARD ---
    st.header("📊 Clinical Analysis Result")
    
    # แถบสรุปตัวเลข (Metrics)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        status_cv = "✅ ปกติ" if cv < 3 else "🟡 เริ่มเสี่ยง" if cv <= 5 else "🔴 เสี่ยงล้มสูง"
        st.metric("Stride Variability", f"{cv:.1f}%", delta=status_cv, delta_color="inverse")
    with m2:
        status_sway = "✅ ปกติ" if 1.5 <= rms_sway <= 2.5 else "🔴 ผิดปกติ"
        st.metric("RMS Trunk Sway", f"{rms_sway:.2f}", delta=status_sway, delta_color="inverse")
    with m3:
        status_power = "✅ แข็งแรง" if sts_power > 300 else "🔴 เสี่ยงกล้ามเนื้อพร่อง" if sts_power < 200 else "🟡 ปกติ"
        st.metric("STS Power", f"{int(sts_power)}W", delta=status_power)
    with m4:
        # จำลอง Fall Risk Index (0-100)
        risk_score = min(100, (cv * 12) + (max(0, rms_sway-2.5) * 15))
        st.metric("Fall Risk Score", f"{int(risk_score)}/100", delta="High Risk" if risk_score > 60 else "Low Risk")

    st.divider()

    # ตารางแปรผลละเอียดพร้อม Reference
    st.subheader("📋 การตีความเชิงคลินิก (Interpretation Table)")
    
    analysis_results = pd.DataFrame({
        "ตัวแปรที่วิเคราะห์": ["Stride Variability (CV%)", "RMS Trunk Sway", "Sit-to-Stand Power"],
        "ค่าที่วัดได้": [f"{cv:.1f} %", f"{rms_sway:.2f}", f"{int(sts_power)} W"],
        "เกณฑ์ปกติ": ["< 3%", "1.5 - 2.5", "> 300 W"],
        "ผลการประเมิน": [
            "🔴 เสี่ยงล้มสูง (Gait Instability)" if cv > 5 else "✅ ปกติ",
            "🔴 การทรงตัวบกพร่อง (Balance Impairment)" if rms_sway > 2.5 else "✅ ปกติ",
            "🔴 เสี่ยงภาวะกล้ามเนื้อพร่อง (Sarcopenia)" if sts_power < 200 else "✅ ปกติ"
        ],
        "อ้างอิงเกณฑ์วัด": ["Prototype Fallsense p.8", "Prototype Fallsense p.8", "Prototype Fallsense p.9"]
    })
    st.table(analysis_results)

    # กราฟ Visualize
    st.subheader("📉 Motion Data Visualization")
    fig = go.Figure()
    # กราฟความเร่ง
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag_filtered'], name="ความเร่ง (Filtered)", line=dict(color="#007AFF", width=2)))
    # จุดที่ตรวจพบก้าว
    fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag_filtered'].iloc[peaks], 
                            mode='markers', name="ตรวจพบจังหวะก้าว", marker=dict(color="#FF3B30", size=10, symbol="x")))
    
    fig.update_layout(
        xaxis_title="เวลา (วินาที)", 
        yaxis_title="ความเร่ง Magnitude (m/s²)", 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ หมายเหตุ: ข้อมูลนี้เป็นผลวิเคราะห์เบื้องต้นจากเซนเซอร์ โปรดปรึกษาแพทย์เพื่อรับการวินิจฉัยอย่างละเอียด")
