import streamlit as st
import joblib
import numpy as np
import pandas as pd

# TIPS: Gunakan @st.cache_resource agar model tidak di-load ulang setiap kali user mengubah input (membuat web lebih cepat)
@st.cache_resource
def load_models():
    # Pastikan nama file .pkl sesuai dengan yang Anda simpan sebelumnya
    model_regresi = joblib.load("burnout_pipeline_GB.pkl") 
    model_klasifikasi = joblib.load("burnout_pipeline_RF_cls.pkl")
    return model_regresi, model_klasifikasi

def main():
    st.title('Student Burnout Prediction')
    st.write("Masukkan data mahasiswa untuk memprediksi tingkat kelelahan (burnout).")

    # --- 1. Load Model ---
    # Pastikan file model Anda (.pkl) ada di folder yang sama atau sesuaikan path-nya
    # model = joblib.load("artifacts/burnout_pipeline_RF.pkl") 
    
    # --- 2. Komponen Input User ---
    # Dikelompokkan menggunakan st.columns agar tampilan tidak terlalu memanjang ke bawah
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informasi Akademik")
        gender = st.radio("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch (Jurusan)", ["CSE", "IT", "ECE", "ME", "CE", "EE", "Other"])
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=0.0)
        tenth_percentage = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=0.0)
        twelfth_percentage = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=0.0)
        attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=0, value=0.0)
        study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=0.0)
        backlogs = st.number_input("Backlogs (Mata kuliah mengulang)", min_value=0, max_value=20, value=0)

    with col2:
        st.subheader("Skill & Aktivitas")
        projects = st.number_input("Projects Completed", min_value=0, max_value=50, value=0)
        internships = st.number_input("Internships Completed", min_value=0, max_value=20, value=0)
        hackathons = st.number_input("Hackathons Participated", min_value=0, max_value=50, value=0)
        certifications = st.number_input("Certifications Count", min_value=0, max_value=50, value=0)
        
        st.write("Rating Skill (1 - 5)")
        coding = st.slider("Coding Skill Rating", 1, 5, 3)
        communication = st.slider("Communication Skill Rating", 1, 5, 3)
        aptitude = st.slider("Aptitude Skill Rating", 1, 5, 3)

    st.divider()
    
    st.subheader("Gaya Hidup & Latar Belakang")
    col3, col4 = st.columns(2)
    
    with col3:
        sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=0.0)
        stress_level = st.number_input("Stress Level" , min_value = 1, max_value=10, value= 0)
        part_time_job = st.radio("Part Time Job", ["No", "Yes"])
        internet_access = st.radio("Internet Access", ["Yes", "No"])
        
    with col4:
        family_income_level = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
        city_tier = st.selectbox("City Tier", ["Tier 3", "Tier 2", "Tier 1"])
        extracurricular = st.selectbox("Extracurricular Involvement", ["None", "Low", "Medium", "High"])

    # --- 3. Membentuk DataFrame ---
    # Kunci dictionary disamakan persis dengan nama kolom pada A.csv
    data = {
        'gender': gender,
        'branch': branch,
        'cgpa': cgpa,
        'tenth_percentage': tenth_percentage,
        'twelfth_percentage': twelfth_percentage,
        'backlogs': backlogs,
        'study_hours_per_day': study_hours,
        'attendance_percentage': attendance_percentage,
        'projects_completed': projects,
        'internships_completed': internships,
        'coding_skill_rating': coding,
        'communication_skill_rating': communication,
        'aptitude_skill_rating': aptitude,
        'hackathons_participated': hackathons,
        'certifications_count': certifications,
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'part_time_job': part_time_job,
        'family_income_level': family_income_level,
        'city_tier': city_tier,
        'internet_access': internet_access,
        'extracurricular_involvement': extracurricular
    }
    
    # Cara membuat DataFrame yang lebih efisien dibandingkan menggunakan list(data.values())
    df = pd.DataFrame([data])

    model_reg, model_cls = load_models()

    # ... [KODE FORM INPUT USER (st.columns, st.radio, dll) TETAP SAMA SEPERTI SEBELUMNYA] ...
    # (Saya potong bagian form di sini agar rapi, letakkan df = pd.DataFrame([data]) di sini)
    
    st.divider()

    # 2. Tombol Prediksi untuk 2 Model
    if st.button("Make Prediction", type="primary", use_container_width=True):
        
        # Lakukan prediksi menggunakan kedua model
        hasil_kategori = model_cls.predict(df)[0] # Hasilnya misal: "Yes" atau 1
        hasil_skor = model_reg.predict(df)[0]     # Hasilnya misal: 7.85
        
        st.subheader("📊 Hasil Analisis")
        
        # Tampilkan hasil secara berdampingan agar terlihat profesional
        col_hasil1, col_hasil2 = st.columns(2)
        
        with col_hasil1:
            # st.metric sangat bagus untuk menampilkan angka/status utama
            st.metric(label="Status Burnout (Klasifikasi)", value=str(hasil_kategori))
            
        with col_hasil2:
            st.metric(label="Tingkat Keparahan (Regresi)", value=f"{hasil_skor:.2f}")
            
        # Opsional: Tambahkan pesan dinamis berdasarkan hasil klasifikasi
        if hasil_kategori == 1 or hasil_kategori == "Yes":
            st.error("Peringatan: Mahasiswa terindikasi mengalami kelelahan mental yang tinggi. Disarankan untuk mengurangi beban SKS atau mengambil waktu istirahat.")
        else:
            st.success("Mahasiswa dalam kondisi aman dan tidak menunjukkan gejala burnout yang signifikan.")

if __name__ == "__main__":
    main()
