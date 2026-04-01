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
    
    # --- กล่องข้อความต้อนรับ ---
    st.info("""
    **GaitPro AI** คือเครื่องมือวิเคราะห์การเดินและการทรงตัวเชิงคลินิก 
    ออกแบบมาเพื่อช่วยคัดกรองความเสี่ยงในการล้มของผู้สูงอายุ โดยอ้างอิงเกณฑ์มาตรฐานจาก **Prototype FallSense**
    """)

    # --- ส่วนที่ 1: ขั้นตอนการเตรียมตัว (ใส่สี Success) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### 🏃‍♂️ 1. การเตรียมตัวและการเดิน (Gait)")
        st.markdown("""
        * **การติดตั้งเซนเซอร์:** ติดอุปกรณ์ที่ **เอวด้านหลัง (L3-L5)** * **ระยะเวลา:** เดินทางตรงปกติเป็นเวลา **10-20 วินาที**
        * **ความเร็ว:** เดินด้วยความเร็วปกติที่เป็นธรรมชาติที่สุด
        * **เป้าหมาย:** เพื่อวัดค่าความแปรปรวนของก้าว (Stride Variability)
        """)

    with col2:
        st.warning("### 🪑 2. การทดสอบลุก-นั่ง (STS)")
        st.markdown("""
        * **อุปกรณ์:** ใช้เก้าอี้มาตรฐานที่ไม่มีที่วางแขน
        * **จำนวน:** ทำการลุกขึ้นและนั่งลงต่อเนื่อง **5 ครั้ง**
        * **การวัด:** ระบบจะคำนวณ **STS Power (W)** * **เป้าหมาย:** เพื่อประเมินความแข็งแรงของกล้ามเนื้อขา
        """)

    st.divider()

    # --- ส่วนที่ 2: การแปรผล (ใส่ตารางและกล่องเน้น) ---
    st.subheader("🧬 เกณฑ์การแปรผลเชิงคลินิก (Clinical Thresholds)")
    
    # สร้างตารางข้อมูลอ้างอิง
    ref_table = {
        "พารามิเตอร์": ["Stride Variability (CV%)", "RMS Trunk Sway", "STS Power"],
        "หน่วย": ["เปอร์เซ็นต์ (%)", "ไม่มีหน่วย", "วัตต์ (W)"],
        "เกณฑ์ปกติ (Safe)": ["น้อยกว่า 3%", "1.5 - 2.5", "มากกว่า 300 W"],
        "เกณฑ์เสี่ยง (Risk)": ["มากกว่า 5%", "มากกว่า 2.5", "น้อยกว่า 200 W"]
    }
    st.table(pd.DataFrame(ref_table))

    # --- ส่วนที่ 3: กล่องข้อความสรุป (ช่วยให้อ่านง่าย) ---
    st.error("""
    **🚨 ข้อควรระวัง:** หากค่า **Stride Variability > 5%** ร่วมกับ **STS Power < 200 W** ผู้ป่วยมีความเสี่ยงสูงที่จะเกิดการล้มในอนาคต ควรได้รับการดูแลเป็นพิเศษ
    """)

    # --- ส่วนที่ 4: เครดิตและเอกสารอ้างอิง ---
    with st.expander("📄 ดูเอกสารอ้างอิงเพิ่มเติม"):
        st.write("""
        * **หน้า 8:** เกณฑ์การวัด Stride Variability และความหมายของ RMS Sway
        * **หน้า 9:** สูตรการคำนวณ Sit-to-Stand Power และเกณฑ์ประเมินภาวะ Sarcopenia
        * **Placement:** การติดเซนเซอร์ระดับ Center of Mass (COM) อ้างอิงหน้า 2
        """)

# --- 4. TAB: ANALYSIS DASHBOARD ---
with tab_analysis:
    if not uploaded_file:
        st.info("👋 กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการวิเคราะห์")
        st.stop()

    try:
        # อ่านไฟล์
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        
        # --- 💡 ส่วนที่แก้ไขใหม่: การจับคู่คอลัมน์แบบยืดหยุ่น ---
        col_map = {}
        for c in df.columns:
            c_low = c.lower()
            if 'timestamp' in c_low: col_map['timestamp'] = c
            if 'ax' in c_low: col_map['ax'] = c
            if 'ay' in c_low: col_map['ay'] = c
            if 'az' in c_low: col_map['az'] = c
        
        # ตรวจเช็คว่าเจอครบไหม
        if len(col_map) < 4:
            st.error(f"❌ คอลัมน์ไม่ครบ! ไฟล์ของคุณมีชื่อคอลัมน์ดังนี้: {list(df.columns)}")
            st.warning("คำแนะนำ: ชื่อคอลัมน์ต้องมีคำว่า timestamp, ax, ay, az ปนอยู่ด้วย (เช่น ax (m/s2))")
            st.stop()
        
        # เปลี่ยนชื่อคอลัมน์ใน DataFrame ให้เป็นชื่อมาตรฐานที่โค้ดเข้าใจ
        df = df.rename(columns={v: k for k, v in col_map.items()})

        # --- ประมวลผลต่อเนื่อง ---
        fs = 50 # จากไฟล์คุณ 0.02s = 50Hz
        df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['mag_f'] = butter_lowpass_filter(df['mag'], cutoff=10, fs=fs)

        # ตรวจจับก้าว
        peaks, _ = find_peaks(df['mag_f'], height=11.0, distance=fs//2)
        stride_times = np.diff(df['timestamp'].iloc[peaks]) if len(peaks) > 1 else []

        cv = (np.std(stride_times) / np.mean(stride_times)) * 100 if len(stride_times) > 0 else 0
        rms_sway = np.sqrt(np.mean(df['mag']**2))
        sts_power = (patient_weight * 9.81 * step_height) / 1.2 

        # --- การแสดงผล ---
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
                "🔴 เสี่ยง" if cv > 5 else "🟢 ปกติ",
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

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
# --- 5. DEEP INTERPRETATION & RECOMMENDATIONS ---
        st.divider()
        st.header("🔍 การตีความผลและการจัดการเชิงลึก (Deep Insights)")

        col_ins1, col_ins2 = st.columns(2)

        with col_ins1:
            st.subheader("🎯 วิเคราะห์ความหมายจากค่าที่วัดได้")
            
            # ตีความค่า Stride Variability
            if cv < 3:
                st.success(f"**Stride Variability ({cv:.1f}%):** อยู่ในเกณฑ์ดีมาก บ่งบอกถึงระบบควบคุมการเคลื่อนไหวของสมองและกล้ามเนื้อทำงานประสานกันได้อย่างสมบูรณ์ (Rhythmic Stability)")
            elif cv <= 5:
                st.warning(f"**Stride Variability ({cv:.1f}%):** เริ่มมีความไม่สม่ำเสมอ อาจเกิดจากความอ่อนล้าของกล้ามเนื้อ หรือเริ่มมีความเสื่อมของระบบประสาทสั่งการ")
            else:
                st.error(f"**Stride Variability ({cv:.1f}%):** อันตราย! การก้าวเท้าแต่ละก้าวสั้นยาวไม่เท่ากันอย่างมาก เสี่ยงต่อการสะดุดหรือเสียหลักล้มได้ง่าย")

            # ตีความค่า RMS Sway
            if 1.5 <= rms_sway <= 2.5:
                st.success(f"**RMS Trunk Sway ({rms_sway:.2f}):** ลำตัวมีความเสถียรขณะเคลื่อนที่ การควบคุมจุดศูนย์ถ่วง (Center of Mass) ทำได้ดี")
            else:
                st.error(f"**RMS Trunk Sway ({rms_sway:.2f}):** มีการเหวี่ยงของลำตัวมากเกินไป บ่งบอกถึงการทรงตัวที่ไม่มั่นคง (Postural Instability) อาจเกิดจากข้อสะโพกหรือกล้ามเนื้อแกนกลางลำตัวไม่แข็งแรง")

        with col_ins2:
            st.subheader("📈 การอ่านกราฟ Motion Waveform")
            st.write("""
            * **ความสูงของยอดคลื่น (Peaks):** บ่งบอกถึงแรงกระแทกขณะลงเท้า (Heel Strike) ยอดที่สม่ำเสมอหมายถึงความแข็งแรงของขาที่เท่ากันทั้งสองข้าง
            * **ระยะห่างระหว่างจุด (Peak Interval):** คือเวลาที่ใช้ในแต่ละก้าว หากระยะห่างนี้กว้างบ้างแคบบ้าง คือสัญญาณของความเสี่ยงล้ม
            * **ความเรียบของเส้น (Signal Smoothness):** เส้นกราฟที่มีสัญญาณรบกวน (Noise) มากเกินไปแม้จะกรองแล้ว อาจหมายถึงอาการสั่น (Tremor) ขณะเดิน
            """)

        st.divider()

        # --- ส่วนแนวทางการจัดการ (Management Plan) ---
        st.subheader("🛠 แนวทางการดูแลและการจัดการต่อไป (Care Plan)")
        
        if cv > 3 or rms_sway > 2.5 or sts_power < 300:
            st.warning("🚨 **ข้อแนะนำสำหรับการดูแลผู้ป่วยรายนี้:**")
            
            plan_col1, plan_col2 = st.columns(2)
            with plan_col1:
                st.markdown("""
                **1. ด้านกายภาพบำบัด (Physical Therapy):**
                * **ฝึกการทรงตัว (Balance Training):** เช่น การยืนขาเดียว หรือเดินต่อเท้า (Tandem Walk)
                * **เพิ่มความแข็งแรง (Strengthening):** เน้นกล้ามเนื้อต้นขา (Quadriceps) เพื่อเพิ่มค่า STS Power
                """)
            with plan_col2:
                st.markdown("""
                **2. ด้านสภาพแวดล้อม (Environment):**
                * เพิ่มราวจับในห้องน้ำและทางเดิน
                * ตรวจสอบแสงสว่างในบ้านให้เพียงพอ
                * หลีกเลี่ยงพื้นผิวที่ลื่นหรือพรมที่ไม่ได้ยึดติดพื้น
                """)
        else:
            st.success("✨ **ผู้ป่วยมีความเสี่ยงต่ำ:** แนะนำให้รักษามาตรฐานกิจกรรมทางกายอย่างสม่ำเสมอ เช่น การเดินเร็ววันละ 20-30 นาที")

        # ปุ่ม Export PDF (จำลอง)
        st.button("🖨 พิมพ์รายงานสรุปผลทางการแพทย์ (PDF)")
