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

# -----------------------------
# Streamlit Page Setup & Theme
# -----------------------------
st.set_page_config(
    page_title="AI-Assisted PCOS Diagnostic System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .stButton>button {background-color:#4CAF50;color:white;height:3em;width:100%;font-size:16px;border-radius:10px;}
    .stMarkdown h1 {color:#FF4B4B;font-size:36px;}
    .stMarkdown h2 {color:#4CAF50;}
    .stTabs [role="tab"] {background-color:#E3F2FD;border-radius:8px;margin:2px;}
    .stDataFrame {border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

st.title("💫 AI-Assisted PCOS Diagnostic & Support System")
st.markdown("""
This Explainable AI system assists doctors in **early detection**, **decision support**, and **patient communication** for PCOS cases.
It identifies subtle hormonal and lifestyle patterns to support accurate, data-driven decisions.
""")

# -----------------------------
# 1️⃣ Load & Train Model
# -----------------------------
@st.cache_resource
def train_ml_model():
    df = pd.read_csv("pcos_dataset.csv")
    df.columns = df.columns.str.strip()

    features = [
        "Age (yrs)", "BMI", "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
        "Cycle(R/I)", "Weight gain(Y/N)", "hair growth(Y/N)", "Reg.Exercise(Y/N)"
    ]
    features = [f for f in features if f in df.columns]
    target_col = "PCOS (Y/N)"

    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found in dataset!")
        st.stop()

    X = df[features].copy()
    y = df[target_col].copy()

    le_dict = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    y_le = LabelEncoder()
    y = y_le.fit_transform(y.astype(str))

    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    return model, le_dict, y_le, features, accuracy, feature_importance

with st.spinner("🔍 Training model..."):
    model, le_dict, y_le, used_features, accuracy, feature_importance = train_ml_model()
st.success(f"🎯 Model trained successfully! Accuracy: {accuracy:.2f}")

# -----------------------------
# Pre-generated AI explanations (hackathon demo)
# -----------------------------
demo_explanations = {
    "Likely to have PCOS": "Based on your inputs, there are signs suggesting PCOS. Lifestyle changes and medical consultation are recommended.",
    "Unlikely to have PCOS": "Your inputs suggest low risk for PCOS. Maintain healthy habits and follow-up as needed."
}

# -----------------------------
# Tabs for Interface
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🧬 Patient Input", "🔮 Prediction Result", "📊 Prediction History"])

# -----------------------------
# 2️⃣ Patient Input Tab
# -----------------------------
with tab1:
    st.header("Enter Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (yrs)", min_value=10, max_value=60, value=25)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=34.0)
        fsh = st.number_input("FSH (mIU/mL)", min_value=0.1, max_value=25.0, value=16.0)
        lh = st.number_input("LH (mIU/mL)", min_value=0.1, max_value=25.0, value=1.0)
    with col2:
        cycle = st.selectbox("Cycle Regularity (R/I)", ["R", "I"])
        with st.expander("Lifestyle Info"):
            exercise = st.selectbox("Regular Exercise (Y/N)", ["Yes", "No"])
            weight_gain = st.selectbox("Recent Weight Gain (Y/N)", ["Yes", "No"])
            hair_growth = st.selectbox("Hair Growth (Y/N)", ["Yes", "No"])

    fsh_lh_ratio = lh / fsh if fsh != 0 else 0

    user_data = pd.DataFrame([{
        "Age (yrs)": age, "BMI": bmi, "FSH(mIU/mL)": fsh, "LH(mIU/mL)": lh,
        "FSH/LH": fsh_lh_ratio, "Cycle(R/I)": cycle,
        "Weight gain(Y/N)": weight_gain, "hair growth(Y/N)": hair_growth,
        "Reg.Exercise(Y/N)": exercise
    }])

    categorical_cols = ["Cycle(R/I)", "Weight gain(Y/N)", "hair growth(Y/N)", "Reg.Exercise(Y/N)"]
    for col in categorical_cols:
        if col in le_dict:
            le = le_dict[col]
            user_data[col] = le.transform(user_data[col].astype(str))
        else:
            mapping = {"R": 0, "I": 1} if col=="Cycle(R/I)" else {"No":0, "Yes":1}
            user_data[col] = user_data[col].map(mapping)

# -----------------------------
# 3️⃣ Prediction & AI Explanation Tab
# -----------------------------
with tab2:
    st.subheader("Predict & AI Explanation")
    if st.button("Predict & Generate PDF"):
        try:
            prediction = model.predict(user_data)[0]
            probability = model.predict_proba(user_data)[0][1]

            threshold = 0.35
            if probability >= threshold:
                prediction_text = "Likely to have PCOS"
                st.error(f"{prediction_text} (Confidence: {probability*100:.1f}%)")
            else:
                prediction_text = "Unlikely to have PCOS"
                st.success(f"{prediction_text} (Confidence: {(1-probability)*100:.1f}%)")

            # Feature Importance
            top_features = feature_importance.head(3)
            st.markdown("### 🧠 Top Features Influencing Prediction")
            for _, row in top_features.iterrows():
                st.write(f"- {row['Feature']} (Importance: {row['Importance']:.2f})")

            fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', color='Importance')
            st.plotly_chart(fig, use_container_width=True)

            # Use pre-generated explanation for demo
            explanation = demo_explanations.get(prediction_text, "AI explanation unavailable.")
            st.subheader("🤖 AI Explanation")
            st.info(explanation)

            # Make text PDF-safe
            safe_prediction_text = prediction_text.encode('latin-1', 'replace').decode('latin-1')
            safe_explanation = explanation.encode('latin-1', 'replace').decode('latin-1')

            # PDF Export
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(0, 12, "PCOS Prediction Report", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Arial", '', 12)
            for key, value in user_data.to_dict(orient='records')[0].items():
                pdf.cell(0, 8, f"{key}: {value}", ln=True)
            pdf.ln(5)
            pdf.cell(0, 10, f"Prediction: {safe_prediction_text} (Confidence: {probability*100:.1f}%)", ln=True)
            pdf.ln(5)
            pdf.multi_cell(0, 10, f"AI Explanation: {safe_explanation}")
            filename = "PCOS_Prediction_Report.pdf"
            pdf.output(filename)
            with open(filename, "rb") as f:
                st.download_button("📄 Download PDF", f, file_name=filename, mime="application/pdf")

            # Save History
            history_df = pd.DataFrame([{
                "Age": age, "BMI": bmi, "FSH": fsh, "LH": lh, "Cycle": cycle,
                "Weight Gain": weight_gain, "Hair Growth": hair_growth,
                "Exercise": exercise, "Prediction": prediction_text, "Probability": probability
            }])
            if os.path.exists("predictions.csv"):
                history_df.to_csv("predictions.csv", mode='a', header=False, index=False)
            else:
                history_df.to_csv("predictions.csv", index=False)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# -----------------------------
# 4️⃣ Prediction History Tab
# -----------------------------
with tab3:
    st.header("📊 Past Predictions")
    if os.path.exists("predictions.csv"):
        df_history = pd.read_csv("predictions.csv")
        st.dataframe(df_history)
    else:
        st.info("No predictions yet.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
### 🩺 How This AI System Helps Doctors
- Pattern Identification: Flags early hormonal or lifestyle imbalances
- Decision Support: Suggests possible PCOS cases for verification with lab tests
- Explainable Reasoning: Shows top features influencing prediction
- Long-Term Tracking: Can monitor trends over time
- Patient Communication: Clear visuals and downloadable report
""")
st.caption("🚀 Developed by Monika | OpenAI Hackathon 2025 | Empowering Women’s Health with AI")
