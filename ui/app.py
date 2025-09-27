import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("final_pipeline.pkl")

model = load_model()

st.title("Heart Disease Prediction")

st.write("Fill the form below to predict whether the patient has heart disease.")

# ------------------------
# Choices mapping
# ------------------------
sex_options = {"Male": 1, "Female": 0}
cp_options = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4
}
fbs_options = {"≤ 120 mg/dl": 0, "> 120 mg/dl": 1}
restecg_options = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}
exang_options = {"No": 0, "Yes": 1}
slope_options = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}
thal_options = {
    "Normal": 3,
    "Fixed defect": 6,
    "Reversible defect": 7
}

# ------------------------
# Form
# ------------------------
with st.form("patient_form"):
    age = st.number_input("Age", 18, 100, 55)
    sex = st.selectbox("Sex", list(sex_options.keys()))
    cp = st.selectbox("Chest pain type (cp)", list(cp_options.keys()))
    trestbps = st.number_input("Resting blood pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Serum cholesterol (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting blood sugar", list(fbs_options.keys()))
    restecg = st.selectbox("Resting ECG (restecg)", list(restecg_options.keys()))
    thalach = st.number_input("Max heart rate achieved (thalach)", 60, 220, 150)
    exang = st.selectbox("Exercise-induced angina (exang)", list(exang_options.keys()))
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope (peak exercise ST segment)", list(slope_options.keys()))
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3], index=0)
    thal = st.selectbox("Thalassemia (thal)", list(thal_options.keys()))

    submitted = st.form_submit_button("Predict")

# ------------------------
# Prediction
# ------------------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex_options[sex],
        "cp": cp_options[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_options[fbs],
        "restecg": restecg_options[restecg],
        "thalach": thalach,
        "exang": exang_options[exang],
        "oldpeak": oldpeak,
        "slope": slope_options[slope],
        "ca": ca,
        "thal": thal_options[thal]
    }])

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0, 1]

    st.subheader("Result")
    if pred == 1:
        st.error(f"Prediction: Disease (probability {proba:.2f})")
    else:
        st.success(f"Prediction: No Disease (probability {proba:.2f})")
