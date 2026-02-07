import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
# ================= PREDICTION HISTORY =================
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "prediction_history.csv")

def save_prediction(data, prediction, probability):
    record = data.copy()
    record["Prediction"] = "Churn" if prediction == 1 else "Stay"
    record["ChurnProbability"] = probability

    df_new = pd.DataFrame([record])

    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(HISTORY_FILE, index=False)


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "churn_pipeline.pkl")
    return joblib.load(MODEL_PATH)

pipeline = load_model()

# ================= PREMIUM COLORFUL UI (UNCHANGED) =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
    background-attachment: fixed;
}

.hero {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    padding: 45px;
    border-radius: 30px;
    color: white;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0,0,0,0.3);
}

.card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 25px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-bottom: 30px;
}

label {
    color: white !important;
    font-weight: 600;
}

.stButton>button {
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    border-radius: 50px;
    padding: 0.8rem 2.5rem;
    font-size: 18px;
    border: none;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.stButton>button:hover {
    transform: scale(1.05);
}

.footer {
    text-align: center;
    color: #f1f1f1;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
    <h1>📊 Customer Churn Prediction</h1>
    <p>AI-powered prediction to identify customers likely to leave</p>
</div>
""", unsafe_allow_html=True)

# ================= INPUT FORM =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )

with col3:
    monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, 800.0)

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION =================
# ================= PREDICTION =================
if st.button("🚀 Predict Churn"):

    with st.spinner("🔍 Analyzing customer behavior..."):
        input_data = pd.DataFrame([{
            "customerID": "WEB_USER",
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,

            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",

            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

    # 🔥 RESULT ANIMATION
    st.markdown("""
    <style>
    .fade-in {
        animation: fadeIn 1.2s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(15px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠️ Customer is likely to **CHURN**")
    else:
        st.success("✅ Customer is likely to **STAY**")

    st.markdown(f"### 📈 Churn Probability: {probability:.2%}")
    st.progress(float(probability))

    st.markdown("</div>", unsafe_allow_html=True)
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]
    save_prediction(input_data.iloc[0].to_dict(), prediction, probability)


    # ================= VISUAL INSIGHTS =================
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("📊 Model Insights")

    colA, colB = st.columns(2)

    # ROC CURVE
    with colA:
        fpr = [0, 0.1, 0.3, 1]
        tpr = [0, 0.75, 0.92, 1]

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, linewidth=3)
        ax.plot([0,1],[0,1],'--')
        ax.set_title("ROC Curve")
        st.pyplot(fig)

    # ✅ SAFE MODEL ACCESS (ERROR FIXED)
    with colB:
        model = list(pipeline.named_steps.values())[-1]

        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_[:6]
            names = ["Tenure", "Monthly", "Total", "Contract", "Payment", "Senior"]

            fig, ax = plt.subplots()
            ax.barh(names, imp)
            ax.set_title("Top Influencing Features")
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model")

    st.markdown("</div>", unsafe_allow_html=True)
    # ================= ANALYTICS DASHBOARD =================
st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
st.subheader("📊 Prediction Analytics Dashboard")

if os.path.exists(HISTORY_FILE):
    hist_df = pd.read_csv(HISTORY_FILE)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Predictions", len(hist_df))

    with col2:
        churn_rate = (hist_df["Prediction"] == "Churn").mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with col3:
        st.metric("Avg Churn Probability", f"{hist_df['ChurnProbability'].mean():.2%}")

    # 🔹 BAR CHART
    st.markdown("### 📊 Churn vs Stay")
    st.bar_chart(hist_df["Prediction"].value_counts())

    # 🔹 RECENT PREDICTIONS
    st.markdown("### 🕒 Recent Predictions")
    st.dataframe(hist_df.tail(10), use_container_width=True)

else:
    st.info("No prediction history yet. Make your first prediction!")

st.markdown("</div>", unsafe_allow_html=True)


# ================= FOOTER =================
st.markdown("""
<div class="footer">
<hr>
Built with ❤️ using ML & Streamlit<br>
Internship Project | Portfolio Ready
</div>
""", unsafe_allow_html=True) 