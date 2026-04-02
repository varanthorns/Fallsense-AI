import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, butter, lfilter

# --- 1. SETTINGS ---
st.set_page_config(page_title="Fallsense AI | Clinical Analysis", layout="wide")

def butter_lowpass_filter(data, cutoff=20, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🩺 Fallsense AI")
    st.write("---")
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ข้อมูล (CSV)", type="csv")
    st.write("---")
    patient_weight = st.number_input("น้ำหนักตัว (kg)", value=70.0)
    step_height = st.number_input("ระยะยกตัวแนวดิ่ง (m)", value=0.45)

tab_analysis, tab_manual = st.tabs(["📊 Analysis Dashboard", "📖 User Manual & References"])

# --- 3. TAB: USER MANUAL & REFERENCES ---
with tab_manual:
    st.header("📖 คู่มือการใช้งานระบบ Fallsense AI")
    
    # --- กล่องข้อความต้อนรับ ---
    st.info("""
    **Fallsense AI** คือเครื่องมือวิเคราะห์การเดินและการทรงตัวเชิงคลินิก 
    ออกแบบมาเพื่อช่วยคัดกรองความเสี่ยงในการล้มของผู้สูงอายุ โดยอ้างอิงเกณฑ์มาตรฐานจาก **Prototype FallSense**
    """)

    # --- ส่วนที่ 1: ขั้นตอนการเตรียมตัว (ใส่สี Success) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### 🏃‍♂️ 1. การเตรียมตัวและการเดิน (Gait)")
        st.markdown("""
        * **การติดตั้งเซนเซอร์:** ติดอุปกรณ์ที่ **เอวด้านหลัง (L3-L5)** 
        * **ระยะเวลา:** เดินทางตรงปกติเป็นเวลา **10-20 วินาที**
        * **ความเร็ว:** เดินด้วยความเร็วปกติที่เป็นธรรมชาติที่สุด
        * **เป้าหมาย:** เพื่อวัดค่าความแปรปรวนของก้าว (Stride Variability)
        """)

    with col2:
        st.warning("### 🪑 2. การทดสอบลุก-นั่ง (STS)")
        st.markdown("""
        * **อุปกรณ์:** ใช้เก้าอี้มาตรฐานที่ไม่มีที่วางแขน
        * **จำนวน:** ทำการลุกขึ้นและนั่งลงต่อเนื่อง **5 ครั้ง**
        * **การวัด:** ระบบจะคำนวณ **STS Power (W)** 
        * **เป้าหมาย:** เพื่อประเมินความแข็งแรงของกล้ามเนื้อขา
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
    with st.expander("📄 รายการเอกสารอ้างอิง (References)"):
        st.markdown("""
        **1. ด้านความแปรปรวนของการเดิน (Stride Variability):**
        * Hausdorff, J. M. (2005). Gait variability: Methods, modeling and meaning. *Journal of NeuroEngineering and Rehabilitation, 2*(1), 1-9. https://doi.org/10.1186/1743-0003-2-19
        
        **2. ด้านการทรงตัวและการเหวี่ยงของลำตัว (RMS Trunk Sway):**
        * Moe-Nilssen, R., & Helbostad, J. L. (2004). Estimation of gait cycle characteristics by trunk accelerometry. *Journal of Biomechanics, 37*(1), 121-126. https://doi.org/10.1016/S0021-9290(03)00233-1
        
        **3. ด้านกำลังกล้ามเนื้อขาและการลุก-นั่ง (STS Power):**
        * Alcazar, J., Losa-Reyna, J., Rodriguez-Lopez, C., Alfaro-Acha, A., Rodriguez-Mañas, L., Ara, I., & Garcia-Garcia, F. J. (2018). The sit-to-stand muscle power test: An easy, inexpensive and portable tool to assess muscle power in older adults. *Experimental Gerontology, 112*, 38-43. https://doi.org/10.1016/j.exger.2018.08.006
        
        **4. ด้านเกณฑ์การประเมินความเสี่ยงล้ม (Fall Risk Thresholds):**
        * Lord, S. R., Sherrington, C., Menz, H. B., & Close, J. C. (2007). *Falls in older people: Risk factors and strategies for prevention* (2nd ed.). Cambridge University Press.
        """)

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
        # =========================================================
        # 5. AI FALL RISK SCORING & INTERPRETATION
        # =========================================================
        st.divider()
        st.header("🧠 Fallsense AI Intelligence Prediction")

        # คำนวณคะแนนความเสี่ยง (Composite Fall Risk Score)
        # ถ่วงน้ำหนัก: CV (40%), Sway (30%), Power (30%)
        cv_score = min(cv * 8, 40)             # เกณฑ์เสี่ยงที่ >5% จะได้คะแนนเต็มในส่วนนี้
        sway_score = min(rms_sway * 12, 30)    # เกณฑ์เสี่ยงที่ >2.5 จะได้คะแนนเต็มในส่วนนี้
        power_penalty = max(0, (300 - sts_power) / 10) # บทลงโทษถ้าแรงน้อยกว่า 300W
        
        total_risk_score = min(100, cv_score + sway_score + power_penalty)

        # การแสดงผลแถบสถานะความเสี่ยง
        risk_col1, risk_col2 = st.columns([1, 2])
        with risk_col1:
            st.metric("Overall Fall Risk Score", f"{total_risk_score:.1f}%")
        with risk_col2:
            if total_risk_score < 35:
                st.success("✅ **ความเสี่ยงต่ำ (Low Risk)**: ร่างกายมีความเสถียรและกำลังกล้ามเนื้ออยู่ในเกณฑ์ดี")
            elif total_risk_score < 65:
                st.warning("⚠️ **ความเสี่ยงปานกลาง (Moderate Risk)**: เริ่มพบความไม่สม่ำเสมอในการเดิน ควรเฝ้าระวัง")
            else:
                st.error("🚨 **ความเสี่ยงสูง (High Risk)**: พบความผิดปกติชัดเจนในหลายดัชนี แนะนำให้ปรึกษาแพทย์")

        # =========================================================
        # 6. CARE PLAN & MANAGEMENT
        # =========================================================
        st.divider()
        st.subheader("🛠 แนวทางการดูแลและการจัดการ (Personalized Care Plan)")
        
        plan_c1, plan_c2 = st.columns(2)
        with plan_c1:
            st.markdown("**🎯 วิเคราะห์ความหมาย:**")
            if cv > 5:
                st.write("- **Gait:** พบการเดินที่ไม่สม่ำเสมอ (Gait Variability) สูง ซึ่งสัมพันธ์กับความเสี่ยงการล้มในผู้สูงอายุ (Hausdorff, 2005)")
            if sts_power < 200:
                st.write("- **Strength:** กำลังกล้ามเนื้อขาต่ำกว่าเกณฑ์ อาจเสี่ยงภาวะกล้ามเนื้อพร่อง (Alcazar et al., 2018)")
            if 1.5 > rms_sway or rms_sway > 2.5:
                st.write("- **Balance:** การทรงตัวไม่นิ่ง มีการแกว่งของลำตัวมากผิดปกติ (Moe-Nilssen, 2004)")

        with plan_c2:
            st.markdown("**📋 คำแนะนำ:**")
            if total_risk_score > 40:
                st.write("1. **ฝึกการทรงตัว:** แนะนำท่าเดินต่อเท้า (Tandem Walk) หรือยืนขาเดียว")
                st.write("2. **เพิ่มแรงขา:** ฝึกลุก-นั่ง (Squat) อย่างสม่ำเสมอวันละ 10-15 ครั้ง")
                st.write("3. **สิ่งแวดล้อม:** ติดตั้งราวจับและเพิ่มแสงสว่างในจุดเสี่ยงของบ้าน")
            else:
                st.write("1. **รักษามาตรฐาน:** ออกกำลังกายแบบแอโรบิก (เดินเร็ว) ต่อเนื่อง")
                st.write("2. **ติดตามผล:** ตรวจสอบความเสี่ยงด้วย Fallsense AI ทุกๆ 1 เดือน")

        # =========================================================
        # 7. EXPORT & RAW DATA
        # =========================================================
        st.divider()
        
        # ปุ่ม Download Report
        report_df = pd.DataFrame({
            "Metric": ["Gait CV", "RMS Sway", "STS Power", "Risk Score"],
            "Value": [f"{cv:.2f}%", f"{rms_sway:.2f}", f"{sts_power:.2f}W", f"{total_risk_score:.2f}%"]
        })
        st.download_button(
            label="📥 ดาวน์โหลดรายงานสรุปผล (CSV)",
            data=report_df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"Fallsense_AI_Report.csv",
            mime="text/csv"
        )

        # รายละเอียดสัญญาณดิบ (เพื่อความโปร)
        with st.expander("🔍 ตรวจสอบสัญญาณเซนเซอร์ดิบ (Raw Data Quality)"):
            raw_fig = go.Figure()
            raw_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ax'], name="Side-to-Side (ax)"))
            raw_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ay'], name="Front-to-Back (ay)"))
            raw_fig.add_trace(go.Scatter(x=df['timestamp'], y=df['az'], name="Vertical (az)"))
            raw_fig.update_layout(title="Tri-axial Raw Acceleration", xaxis_title="Time (s)", yaxis_title="m/s²")
            st.plotly_chart(raw_fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")

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
