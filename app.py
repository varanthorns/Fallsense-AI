import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, welch

# --- 1. Page Configuration & Theme ---
st.set_page_config(
    page_title="GaitPro AI | Clinical Motion Analysis",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Medical Look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_value=True)

# --- 2. Sidebar & Navigation ---
with st.sidebar:
    st.title("🩺 GaitPro AI™")
    st.subheader("System Control")
    
    menu = st.radio("Navigation", ["Home / Dashboard", "User Manual (วิธีใช้งาน)", "Patient History"])
    
    st.divider()
    uploaded_file = st.file_uploader("📂 Upload CSV Data", type="csv")
    
    if uploaded_file:
        st.success("Data Loaded Successfully!")
        patient_id = st.text_input("Patient ID", "PT-2024-001")
        weight = st.number_input("Weight (kg)", 30, 150, 70)
        sensitivity = st.slider("Step Sensitivity", 5.0, 15.0, 10.5)

# --- 3. PAGE: User Manual (วิธีใช้งานอย่างละเอียด) ---
if menu == "User Manual (วิธีใช้งาน)":
    st.header("📖 คู่มือการใช้งานระบบ GaitPro AI™ อย่างละเอียด")
    
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("1️⃣ การเตรียมอุปกรณ์ (Hardware)", expanded=True):
            st.write("""
            * **ตำแหน่งติดตั้ง:** ติดกล่องบริเวณ **กึ่งกลางบั้นเอว (L5/S1 Vertebrae)** * **ทิศทาง:** ให้เซนเซอร์แนบสนิทกับตัว ไม่แกว่งไปมาขณะเดิน
            * **การเปิดเครื่อง:** เปิดสวิตช์ รอจนไฟ Wi-Fi ติดนิ่ง
            """)
        
        with st.expander("2️⃣ การเก็บข้อมูล (Recording)", expanded=True):
            st.write("""
            * เชื่อมต่อ Wi-Fi ชื่อ **'Fallsense_AP'** ผ่าน iPad/MacBook
            * เข้าไปที่ URL: `192.168.4.1`
            * กดปุ่ม **[Record]** -> ให้ผู้ป่วยเดิน 10-20 เมตร -> กด **[Stop]**
            """)
            
    with col_b:
        with st.expander("3️⃣ การนำข้อมูลเข้า (Importing)", expanded=True):
            st.write("""
            * กดปุ่ม **[💾 Download CSV]** ในหน้าเว็บเพื่อเซฟไฟล์ลงเครื่อง
            * กลับมาที่แอปนี้ (Streamlit) แล้วลากไฟล์ `data.csv` ใส่ในช่อง Sidebar ด้านซ้าย
            """)
            
        with st.expander("4️⃣ การอ่านผลวิเคราะห์ (Interpretation)", expanded=True):
            st.write("""
            * **Total Steps:** จำนวนก้าวทั้งหมดที่ตรวจพบ
            * **Stride CV:** ความแปรปรวน (ถ้า > 5% คือเสี่ยงล้มสูง)
            * **Frequency:** จังหวะการเดินปกติควรอยู่ที่ 1.6 - 2.2 Hz
            """)
    st.info("💡 **Tips:** หากระบบนับก้าวไม่ตรง ให้ปรับ 'Step Sensitivity' ที่แถบด้านซ้าย")

# --- 4. PAGE: Home / Dashboard ---
elif menu == "Home / Dashboard":
    if not uploaded_file:
        st.title("Welcome to GaitPro AI Dashboard")
        st.image("https://static.vecteezy.com/system/resources/previews/004/578/713/original/human-walking-cycle-animation-free-vector.jpg", width=600)
        st.warning("กรุณาอัปโหลดไฟล์ CSV จากแถบด้านซ้ายเพื่อเริ่มการวิเคราะห์")
    else:
        # --- DATA PROCESSING ---
        df = pd.read_csv(uploaded_file)
        df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['mag_smooth'] = df['mag'].rolling(window=5).mean() # Noise reduction
        
        # Peak Detection
        peaks, _ = find_peaks(df['mag_smooth'], height=sensitivity, distance=25)
        step_count = len(peaks)
        
        # Gait Metrics
        stride_times = np.diff(df['timestamp'].iloc[peaks]) if step_count > 1 else [0]
        cv = (np.std(stride_times) / np.mean(stride_times)) * 100 if len(stride_times) > 0 else 0
        
        # FFT Analysis
        try:
            fs = 1 / (df['timestamp'].iloc[1] - df['timestamp'].iloc[0])
            f, psd = welch(df['mag'].dropna(), fs, nperseg=min(len(df), 256))
            dom_freq = f[np.argmax(psd)]
        except:
            dom_freq = 0

        # --- DASHBOARD UI ---
        st.subheader(f"📊 Analysis Report: {patient_id}")
        
        # Row 1: Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Steps", f"{step_count}")
        m2.metric("Stride CV (Variability)", f"{cv:.1f}%", delta="-3.2%" if cv < 5 else "+5.1%", delta_color="inverse")
        m3.metric("Cadence (steps/min)", f"{round(step_count/(df['timestamp'].max()/60),1)}")
        
        risk_score = min(100, (cv * 12) + (abs(dom_freq - 1.8) * 8))
        m4.metric("Clinical Risk Index", f"{risk_score:.1f}/100")

        # Row 2: Charts
        st.divider()
        chart_tab, freq_tab = st.tabs(["📉 Motion Analysis (Time Domain)", "🧬 Stability Analysis (FFT)"])
        
        with chart_tab:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=("Acceleration Magnitude (G)", "Angular Velocity (deg/s)"))
            # Acceleration
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag_smooth'], name="Accel Mag", line=dict(color='#1E88E5')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag_smooth'].iloc[peaks], mode='markers', name='Heel Strike', marker=dict(color='red', size=10)), row=1, col=1)
            # Gyro
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['gx'], name="Gyro X", line=dict(color='#FFC107')), row=2, col=1)
            fig.update_layout(height=500, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with freq_tab:
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=f, y=psd, fill='tozeroy', name="Power Spectral Density", line=dict(color='#673AB7')))
            fig_fft.update_layout(title="Rhythm Stability (Dominant Frequency)", xaxis_title="Frequency (Hz)", yaxis_title="Power", height=400)
            st.plotly_chart(fig_fft, use_container_width=True)

        # Row 3: Medical Conclusion
        st.divider()
        res_col, act_col = st.columns([2, 1])
        with res_col:
            st.subheader("👨‍⚕️ Clinical Assessment")
            if risk_score > 60 or cv > 5:
                st.error(f"**High Fall Risk Detected!** (CV: {cv:.1f}%)\n\nผู้ป่วยมีจังหวะการเดินที่ไม่สม่ำเสมออย่างรุนแรง เสี่ยงต่อการล้มสูง ควรพบแพทย์เพื่อตรวจระบบประสาทและกล้ามเนื้อ")
            else:
                st.success(f"**Gait Stability: Normal** (CV: {cv:.1f}%)\n\nผู้ป่วยมีการเดินที่สม่ำเสมอและมีความมั่นคงสูง")
        
        with act_col:
            st.subheader("📥 Actions")
            st.button("Generate PDF Report")
            st.button("Save to EHR Database")

# --- 5. PAGE: Patient History ---
elif menu == "Patient History":
    st.header("📂 Patient Data History")
    st.info("ฟีเจอร์นี้จะเชื่อมต่อกับฐานข้อมูลในอนาคตเพื่อดูแนวโน้มการเดินของผู้ป่วยรายบุคคล")
    st.table(pd.DataFrame({
        "Date": ["2024-03-01", "2024-03-15", "2024-04-01"],
        "Risk Score": [75.2, 60.1, 42.5],
        "Status": ["High Risk", "Moderate", "Low Risk"]
    }))
