# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from fpdf import FPDF
import os
from datetime import datetime

# -----------------------------
# Configuration & Theme
# -----------------------------
st.set_page_config(page_title="AI-Assisted PCOS Diagnostic System",
                   page_icon="💊", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {background-color: #f8fbfd;}
    .stButton>button {background-color:#1976d2;color:white;height:2.8em;border-radius:8px;font-weight:600;}
    .stDownloadButton>button {background-color:#4CAF50;color:white;height:2.6em;border-radius:8px;}
    .stMarkdown h1 {color:#0d47a1;}
    .stAlert {border-radius:8px;}
    </style>
""", unsafe_allow_html=True)

st.title("AI-Assisted PCOS Diagnostic & Support System")
st.markdown("An explainable AI assistant for early detection, doctor decision support, and patient communication.")

# -----------------------------
# Simple Login System
# -----------------------------
USER_CREDENTIALS = {"doctor": "password123", "admin": "monica"}  
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

def do_logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.rerun()  # updated

if not st.session_state.authenticated:
    st.header("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()  # updated
        else:
            st.error("Invalid username or password")
    st.stop()
else:
    st.sidebar.write(f"Signed in as **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        do_logout()

# -----------------------------
# OpenAI setup
# -----------------------------
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    openai.api_key = st.secrets["openai"]["api_key"]
else:
    openai.api_key = None

# -----------------------------
# AI Explanation Functions
# -----------------------------
def call_openai_for_explanations(user_info, prediction_text, probability):
    prompt = f"""
You are a helpful medical assistant. Given the patient info and prediction below, produce three short sections:
1) Doctor Explanation (2-4 sentences, clinical, mentions which features influenced the prediction),
2) Patient Explanation (2 sentences, very simple, friendly, not technical),
3) Doctor Recommendation (one-line clinical recommendation: tests/next steps).

Patient info:
{user_info}

Prediction: {prediction_text}
Confidence: {probability*100:.1f}%
Return the three sections labeled clearly as DOCTOR_EXPLANATION:, PATIENT_EXPLANATION:, DOCTOR_NOTE:
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"You are a helpful medical assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        doctor_expl, patient_expl, doctor_note = "", "", ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        current = None
        for ln in lines:
            up = ln.upper()
            if up.startswith("DOCTOR_EXPLANATION"):
                current = "doctor"
                content = ln.split(":",1)[1].strip() if ":" in ln else ""
                doctor_expl += content + " "
            elif up.startswith("PATIENT_EXPLANATION"):
                current = "patient"
                content = ln.split(":",1)[1].strip() if ":" in ln else ""
                patient_expl += content + " "
            elif up.startswith("DOCTOR_NOTE"):
                current = "note"
                content = ln.split(":",1)[1].strip() if ":" in ln else ""
                doctor_note += content + " "
            else:
                if current == "doctor": doctor_expl += ln + " "
                elif current == "patient": patient_expl += ln + " "
                elif current == "note": doctor_note += ln + " "
        return doctor_expl.strip(), patient_expl.strip(), doctor_note.strip()
    except Exception:
        return None

def demo_explanations_for(prediction_text, probability, user_info):
    if "Likely" in prediction_text:
        return ("The model found patterns commonly associated with PCOS such as hormonal imbalance and irregular cycles; BMI and LH/FSH ratio influenced the assessment.",
                "There are signs that suggest PCOS. Please consult a doctor for more tests and follow recommended lifestyle changes.",
                "Recommend ultrasound and follow-up tests; consider lifestyle interventions and a 3-month review.")
    else:
        return ("The model did not find strong indicators of PCOS in the provided data. Key features were within expected ranges.",
                "Your results suggest low risk for PCOS. Keep healthy habits and follow up if symptoms change.",
                "Continue routine monitoring and healthy lifestyle; repeat evaluation if symptoms arise.")

demo_explanations = {
    "Likely to have PCOS": demo_explanations_for("Likely to have PCOS", 0.7, {}),
    "Unlikely to have PCOS": demo_explanations_for("Unlikely to have PCOS", 0.2, {})
}

# -----------------------------
# ML Model training
# -----------------------------
@st.cache_resource
def train_ml_model():
    df = pd.read_csv("pcos_dataset.csv", on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    features = ["Age (yrs)", "BMI", "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
                "Cycle(R/I)", "Weight gain(Y/N)", "hair growth(Y/N)", "Reg.Exercise(Y/N)"]
    features = [f for f in features if f in df.columns]
    target_col = "PCOS (Y/N)"
    if target_col not in df.columns: raise ValueError(f"Target '{target_col}' missing!")

    X = df[features].copy()
    y = LabelEncoder().fit_transform(df[target_col].astype(str))

    le_dict = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    X = X.fillna(X.median())
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
    accuracy = model.score(X, y)
    return model, le_dict, features, accuracy, feature_importance

with st.spinner("🔍 Training model..."):
    try:
        model, le_dict, used_features, accuracy, feature_importance = train_ml_model()
    except Exception as e:
        st.error(f"Error loading/training model: {e}")
        st.stop()
st.success(f"Model ready — accuracy (train): {accuracy:.2f}")

# -----------------------------
# App Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🧬 Patient Input", "🔮 Prediction & AI", "📊 Prediction History"])

# --- Patient Input ---
with tab1:
    st.header("Enter Patient Details")
    patient_name = st.text_input("👩 Patient Name")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (yrs)", min_value=10, max_value=100, value=25)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        fsh = st.number_input("FSH (mIU/mL)", min_value=0.0, max_value=100.0, value=6.0, format="%.2f")
        lh = st.number_input("LH (mIU/mL)", min_value=0.0, max_value=100.0, value=8.0, format="%.2f")
    with col2:
        cycle = st.selectbox("Cycle Regularity (R/I)", ["R", "I"])
        with st.expander("Lifestyle Info"):
            exercise = st.selectbox("Regular Exercise (Y/N)", ["Yes", "No"])
            weight_gain = st.selectbox("Recent Weight Gain (Y/N)", ["Yes", "No"])
            hair_growth = st.selectbox("Hair Growth (Y/N)", ["Yes", "No"])
    fsh_lh_ratio = (lh / fsh) if fsh != 0 else 0.0
    user_data = pd.DataFrame([{
        "Age (yrs)": age, "BMI": bmi, "FSH(mIU/mL)": fsh, "LH(mIU/mL)": lh,
        "FSH/LH": fsh_lh_ratio, "Cycle(R/I)": cycle,
        "Weight gain(Y/N)": weight_gain, "hair growth(Y/N)": hair_growth,
        "Reg.Exercise(Y/N)": exercise
    }])
    for col in ["Cycle(R/I)", "Weight gain(Y/N)", "hair growth(Y/N)", "Reg.Exercise(Y/N)"]:
        if col in le_dict: user_data[col] = le_dict[col].transform(user_data[col].astype(str))
        else: user_data[col] = user_data[col].map({"R":0,"I":1,"No":0,"Yes":1})

# --- Prediction & AI ---
with tab2:
    st.subheader("Predict & AI Explanations")
    st.write("Click **Predict** to run the model. Toggle AI explanations on/off below.")
    ai_toggle = st.checkbox("Enable live AI explanations", value=False)

    if st.button("Predict"):
        try:
            # Run prediction
            prediction = model.predict(user_data)[0]
            proba = model.predict_proba(user_data)[0][1]  # probability of positive class
            threshold = 0.6  # realistic threshold
            if proba >= threshold:
                prediction_text = "Likely to have PCOS"
                st.error(f"{prediction_text} — Confidence: {proba*100:.1f}%")
            else:
                prediction_text = "Unlikely to have PCOS"
                st.success(f"{prediction_text} — Confidence: {(1-proba)*100:.1f}%")

            # Show top features
            st.markdown("### 🧠 Top Features Influencing Prediction")
            top_features = feature_importance.head(5).reset_index(drop=True)
            st.dataframe(top_features.style.format({"Importance": "{:.3f}"}))
            fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance')
            st.plotly_chart(fig, use_container_width=True)

            # Prepare patient info for AI
            user_info = {
                "Patient Name": patient_name or "N/A",
                "Age": age, "BMI": bmi, "FSH": fsh, "LH": lh,
                "FSH/LH": round(fsh_lh_ratio, 3),
                "Cycle": cycle, "Weight Gain": weight_gain,
                "Hair Growth": hair_growth, "Exercise": exercise
            }

            # AI Explanations
            doctor_expl, patient_expl, doc_note = None, None, None
            if ai_toggle and openai.api_key:
                result = call_openai_for_explanations(user_info, prediction_text, proba)
                if result:
                    doctor_expl, patient_expl, doc_note = result

            # Fallback to demo explanations if live AI fails
            if not doctor_expl:
                doctor_expl, patient_expl, doc_note = demo_explanations.get(
                    prediction_text, demo_explanations_for(prediction_text, proba, user_info)
                )

            # Display AI outputs
            st.markdown("### 🤖 AI Explanations")
            with st.expander("Doctor Explanation (detailed)"):
                st.write(doctor_expl)
            with st.expander("Patient Explanation (simple)"):
                st.write(patient_expl)
            with st.expander("Doctor's Note / Recommendation"):
                st.write(doc_note)

            # PDF generation
            def generate_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "PCOS Prediction Report", ln=True, align="C")
                pdf.ln(6)
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                pdf.cell(0, 8, f"Patient Name: {patient_name}", ln=True)
                pdf.cell(0, 8, f"Age: {age} | BMI: {bmi}", ln=True)
                pdf.ln(4)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, f"Prediction: {prediction_text} (Confidence: {proba*100:.1f}%)", ln=True)
                pdf.ln(4)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Doctor Explanation:", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 7, doctor_expl)
                pdf.ln(2)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Patient Explanation:", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 7, patient_expl)
                pdf.ln(2)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Doctor's Note / Recommendation:", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 7, doc_note)
                fname = f"PCOS_Report_{(patient_name or 'patient').replace(' ','_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
                pdf.output(fname)
                return fname

            pdf_file = generate_pdf()
            with open(pdf_file, "rb") as f:
                st.download_button("📄 Download Report (PDF)", f, file_name=pdf_file, mime="application/pdf")

            # Save prediction to history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "user": st.session_state.username,
                "patient_name": patient_name,
                "age": age, "bmi": bmi, "fsh": fsh, "lh": lh,
                "fsh_lh": round(fsh_lh_ratio, 3),
                "cycle": cycle, "weight_gain": weight_gain,
                "hair_growth": hair_growth, "exercise": exercise,
                "prediction": prediction_text, "probability": proba
            }
            hist_df = pd.DataFrame([history_entry])
            if os.path.exists("predictions.csv"):
                hist_df.to_csv("predictions.csv", mode='a', header=False, index=False)
            else:
                hist_df.to_csv("predictions.csv", index=False)

            st.success("✅ Prediction complete and saved to history.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")


# --- Prediction History ---
with tab3:
    st.header("📊 Prediction History")
    if os.path.exists("predictions.csv"):
        df_history = pd.read_csv("predictions.csv", on_bad_lines='skip')
        if 'timestamp' in df_history.columns:
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], errors='coerce')
            df_history = df_history.sort_values(by="timestamp", ascending=False)
        st.dataframe(df_history.reset_index(drop=True))
        with open("predictions.csv", "rb") as fh:
            st.download_button("Download full history (CSV)", fh, file_name="predictions.csv", mime="text/csv")
    else:
        st.info("No predictions yet. History will appear here after first prediction.")

st.markdown("---")
st.markdown("""
### 🤖 About This AI-Assisted PCOS Diagnostic System
- This web application leverages **Artificial Intelligence** to provide **early, data-driven insights** for PCOS detection.
- The AI model analyzes patient features such as hormones, BMI, cycle patterns, and lifestyle factors to **assist doctors in making informed decisions**.
- Explanations are provided in **two forms**:
  - **Doctor Explanation:** Detailed clinical reasoning highlighting which factors influenced the prediction.
  - **Patient Explanation:** Simple, friendly explanation to help patients understand their results.
- The system also generates **PDF reports** for easy record-keeping and tracks **prediction history** for ongoing patient management.
- This tool **enhances healthcare efficiency** and empowers doctors and patients with actionable insights, but it is **not a replacement for professional medical advice**.
""")
st.caption("🚀 Developed by MAVericks | SEEKH 2025 | AI-Assisted PCOS Diagnostic System")

