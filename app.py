import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, butter, lfilter

# --- 1. SETTINGS ---
st.set_page_config(page_title="GaitPro AI | Clinical Analysis", layout="wide")

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

tab_analysis, tab_manual = st.tabs(["📊 Analysis Dashboard", "📖 User Manual & References"])

# --- 3. TAB: USER MANUAL & REFERENCES ---
with tab_manual:
    st.header("📖 คู่มือการใช้งานระบบ GaitPro AI")
    st.info("**GaitPro AI** คือเครื่องมือวิเคราะห์การเดินและการทรงตัวเชิงคลินิก ออกแบบมาเพื่อคัดกรองความเสี่ยงในการล้ม")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("### 🏃‍♂️ 1. การเดิน (Gait)\n* ติดเซนเซอร์ที่เอวหลัง (L3-L5)\n* เดินปกติ 10-20 วินาที")
    with col2:
        st.warning("### 🪑 2. ลุก-นั่ง (STS)\n* ลุก-นั่งต่อเนื่อง 5 ครั้ง\n* เพื่อวัดความแข็งแรงกล้ามเนื้อขา")

    st.divider()
    st.subheader("🧬 เกณฑ์การแปรผลเชิงคลินิก")
    ref_table = {
        "พารามิเตอร์": ["Stride Variability (CV%)", "RMS Trunk Sway", "STS Power"],
        "ปกติ (Safe)": ["< 3%", "1.5 - 2.5", "> 300 W"],
        "เสี่ยง (Risk)": ["> 5%", "> 2.5", "< 200 W"]
    }
    st.table(pd.DataFrame(ref_table))

# --- 4. TAB: ANALYSIS DASHBOARD ---
with tab_analysis:
    if not uploaded_file:
        st.info("👋 กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการวิเคราะห์")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        
        # จับคู่คอลัมน์แบบยืดหยุ่น
        col_map = {}
        for c in df.columns:
            c_low = c.lower()
            if 'timestamp' in c_low: col_map['timestamp'] = c
            if 'ax' in c_low: col_map['ax'] = c
            if 'ay' in c_low: col_map['ay'] = c
            if 'az' in c_low: col_map['az'] = c
        
        if len(col_map) < 4:
            st.error(f"❌ คอลัมน์ไม่ครบ! ไฟล์ของคุณมี: {list(df.columns)}")
            st.stop()
        
        df = df.rename(columns={v: k for k, v in col_map.items()})

        # ประมวลผล
        fs = 50 
        df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['mag_f'] = butter_lowpass_filter(df['mag'], cutoff=10, fs=fs)

        peaks, _ = find_peaks(df['mag_f'], height=11.0, distance=fs//2)
        
        # ป้องกันกรณีหา Peak ไม่เจอ
        if len(peaks) > 1:
            stride_times = np.diff(df['timestamp'].iloc[peaks])
            cv = (np.std(stride_times) / np.mean(stride_times)) * 100
        else:
            cv = 0
            
        rms_sway = np.sqrt(np.mean(df['mag']**2))
        sts_power = (patient_weight * 9.81 * step_height) / 1.2 

        # --- การแสดงผลหลัก ---
        st.header("📊 Clinical Analysis Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Stride Variability", f"{cv:.1f}%")
        c2.metric("RMS Trunk Sway", f"{rms_sway:.2f}")
        c3.metric("STS Power", f"{int(sts_power)}W")

        st.divider()
        st.subheader("📋 ตารางประเมินผล")
        res_table = pd.DataFrame({
            "ตัวแปร": ["Stride Var", "RMS Sway", "STS Power"],
            "ค่าที่ได้": [f"{cv:.1f}%", f"{rms_sway:.2f}", f"{int(sts_power)}W"],
            "สถานะ": [
                "🔴 เสี่ยง" if cv > 5 else "🟡 ระวัง" if cv > 3 else "🟢 ปกติ",
                "🔴 ไม่นิ่ง" if rms_sway > 2.5 else "🟢 ปกติ",
                "🔴 ต่ำ" if sts_power < 200 else "🟢 ปกติ"
            ]
        })
        st.table(res_table)

        # กราฟ
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mag_f'], name="Acceleration (Mag)"))
        fig.add_trace(go.Scatter(x=df['timestamp'].iloc[peaks], y=df['mag_f'].iloc[peaks], mode='markers', name="Detected Steps", marker=dict(color="red")))
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. DEEP INTERPRETATION (ย้ายเข้ามาอยู่ใน Try เพื่อความปลอดภัย) ---
        st.divider()
        st.header("🔍 การตีความผลและการจัดการเชิงลึก (Deep Insights)")
        col_ins1, col_ins2 = st.columns(2)

        with col_ins1:
            st.subheader("🎯 วิเคราะห์ความหมายจากค่าที่วัดได้")
            if cv < 3:
                st.success(f"**Stride Variability ({cv:.1f}%):** ดีมาก ระบบประสาทสั่งการประสานงานได้สมบูรณ์")
            elif cv <= 5:
                st.warning(f"**Stride Variability ({cv:.1f}%):** เริ่มไม่สม่ำเสมอ อาจเกิดจากความอ่อนล้า")
            else:
                st.error(f"**Stride Variability ({cv:.1f}%):** อันตราย! เสี่ยงต่อการสะดุดล้มสูง")

            if 1.5 <= rms_sway <= 2.5:
                st.success(f"**RMS Trunk Sway ({rms_sway:.2f}):** การทรงตัวดี จุดศูนย์ถ่วงมั่นคง")
            else:
                st.error(f"**RMS Trunk Sway ({rms_sway:.2f}):** ลำตัวเหวี่ยงมาก เสี่ยงเสียหลักขณะเดิน")

        with col_ins2:
            st.subheader("📈 การอ่านกราฟ Motion Waveform")
            st.write("""
            * **Heel Strike (Peaks):** ยอดคลื่นที่สม่ำเสมอหมายถึงแรงกระแทกเท้าที่เท่ากันสองข้าง
            * **Peak Interval:** ระยะห่างที่กว้างบ้างแคบบ้าง คือสัญญาณความเสี่ยงล้ม
            """)

        st.divider()
        st.subheader("🛠 แนวทางการดูแลและการจัดการต่อไป (Care Plan)")
        if cv > 3 or rms_sway > 2.5 or sts_power < 300:
            st.warning("🚨 **ข้อแนะนำสำหรับการดูแลผู้ป่วยรายนี้:**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("**1. กายภาพบำบัด:** ฝึกยืนขาเดียว (Tandem Walk) และเพิ่มกล้ามเนื้อต้นขา")
            with p_col2:
                st.markdown("**2. สภาพแวดล้อม:** เพิ่มราวจับในบ้าน และตรวจสอบแสงสว่างให้เพียงพอ")
        else:
            st.success("✨ **ผู้ป่วยมีความเสี่ยงต่ำ:** แนะนำกิจกรรมทางกายสม่ำเสมอ เช่น เดินเร็ววันละ 30 นาที")

        st.button("🖨 พิมพ์รายงานสรุปผล (PDF)")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
