import streamlit as st
import joblib
import numpy as np
import pandas as pd

# @st.cache_resource agar model tidak di-load ulang setiap kali user mengubah input
@st.cache_resource
def load_models():
    model_regresi = joblib.load("salary_pipeline_GB.pkl") 
    model_klasifikasi = joblib.load("salary_pipeline_RF_cls.pkl")
    return model_regresi, model_klasifikasi

def main():
    st.title('Employment dan Salary Prediction')
    st.write("Masukkan data mahasiswa untuk memprediksi tingkat burnout.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Akademik")
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch (Jurusan)", ["CSE", "IT", "ECE", "ME", "CE", "EE", "Other"])
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=0.0)
        tenth_percentage = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=0.0)
        twelfth_percentage = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=0.0)
        attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=0.0, value=0.0)
        study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=0.0)
        backlogs = st.number_input("Backlogs (Mata kuliah mengulang)", min_value=0, max_value=20, value=0.0)

    with col2:
        st.subheader("Skill & Aktivitas")
        projects = st.number_input("Projects Completed", min_value=0, max_value=50, value=0)
        internships = st.number_input("Internships Completed", min_value=0, max_value=20, value=0.0)
        hackathons = st.number_input("Hackathons Participated", min_value=0, max_value=50, value=0.0)
        certifications = st.number_input("Certifications Count", min_value=0, max_value=50, value=0.0)
        
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
        part_time_job = st.selectbox("Part Time Job", ["No", "Yes"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        
    with col4:
        family_income_level = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
        city_tier = st.selectbox("City Tier", ["Tier 3", "Tier 2", "Tier 1"])
        extracurricular = st.selectbox("Extracurricular Involvement", ["None", "Low", "Medium", "High"])

    # Mapping DataFrame
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
    df = pd.DataFrame([data])

    model_reg, model_cls = load_models()

    st.divider()
    # Tombol Prediksi
    if st.button("Make Prediction", type="primary", use_container_width=True):

        hasil_kategori = model_cls.predict(df)[0] # Hasilnya misal: "Yes" atau 1
        hasil_skor = model_reg.predict(df)[0]     # Hasilnya misal: 7.85
        
        st.subheader("Hasil Analisis:")
        col_hasil1, col_hasil2 = st.columns(2)
        
        with col_hasil1:
            st.metric(label="Status Employment", value=str(hasil_kategori))
            
        with col_hasil2:
            st.metric(label="Salary (LGPA)", value=f"{hasil_skor:.2f}")
            
if __name__ == "__main__":
    main()
