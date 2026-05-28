import streamlit as st
import pickle
import numpy as np

# Load model, scaler and features
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="💳",
    layout="wide"
)

# Header
st.title("💳 Credit Risk Scoring Model")
st.markdown("**McKinsey-style decision support tool** — Enter a customer profile to get an instant risk assessment.")
st.divider()

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Profile")
    
    credit_limit = st.slider("Credit Limit (NT$)", 
                              min_value=10000, max_value=1000000, 
                              value=200000, step=10000)
    
    age = st.slider("Age", min_value=21, max_value=79, value=35)
    
    gender = st.selectbox("Gender", options=[1, 2], 
                          format_func=lambda x: "Male" if x == 1 else "Female")
    
    education = st.selectbox("Education", options=[1, 2, 3, 4],
                             format_func=lambda x: {
                                 1: "Graduate School", 
                                 2: "University", 
                                 3: "High School", 
                                 4: "Others"
                             }[x])
    
    marital_status = st.selectbox("Marital Status", options=[1, 2, 3],
                                  format_func=lambda x: {
                                      1: "Married", 
                                      2: "Single", 
                                      3: "Others"
                                  }[x])

with col2:
    st.subheader("Payment History (last 6 months)")
    st.caption("Payment status: -2=no consumption, -1=paid in full, 0=revolving credit, 1+=months delayed")
    
    pay_sep = st.slider("September (most recent)", min_value=-2, max_value=8, value=0)
    pay_aug = st.slider("August", min_value=-2, max_value=8, value=0)
    pay_jul = st.slider("July", min_value=-2, max_value=8, value=0)
    pay_jun = st.slider("June", min_value=-2, max_value=8, value=0)
    pay_may = st.slider("May", min_value=-2, max_value=8, value=0)
    pay_apr = st.slider("April", min_value=-2, max_value=8, value=0)

st.divider()
st.subheader("Bill & Payment Amounts")
col3, col4 = st.columns(2)

with col3:
    st.caption("Bill amounts (NT$)")
    bill_sep = st.number_input("Bill Sep", value=50000, step=1000)
    bill_aug = st.number_input("Bill Aug", value=48000, step=1000)
    bill_jul = st.number_input("Bill Jul", value=46000, step=1000)
    bill_jun = st.number_input("Bill Jun", value=44000, step=1000)
    bill_may = st.number_input("Bill May", value=42000, step=1000)
    bill_apr = st.number_input("Bill Apr", value=40000, step=1000)

with col4:
    st.caption("Payment amounts (NT$)")
    pay_amt_sep = st.number_input("Payment Sep", value=2000, step=500)
    pay_amt_aug = st.number_input("Payment Aug", value=2000, step=500)
    pay_amt_jul = st.number_input("Payment Jul", value=2000, step=500)
    pay_amt_jun = st.number_input("Payment Jun", value=2000, step=500)
    pay_amt_may = st.number_input("Payment May", value=2000, step=500)
    pay_amt_apr = st.number_input("Payment Apr", value=2000, step=500)

st.divider()

# Predict button
if st.button("🔍 Assess Credit Risk", type="primary", use_container_width=True):
    
    # Engineer features same as training
    utilisation_ratio = bill_sep / (credit_limit + 1)
    payment_trend = pay_amt_sep - pay_amt_apr
    missed_payments = sum([1 for p in [pay_sep, pay_aug, pay_jul, pay_jun, pay_may, pay_apr] if p > 0])
    
    # Build input array in same order as training
    input_data = np.array([[
        credit_limit, gender, education, marital_status, age,
        pay_sep, pay_aug, pay_jul, pay_jun, pay_may, pay_apr,
        bill_sep, bill_aug, bill_jul, bill_jun, bill_may, bill_apr,
        pay_amt_sep, pay_amt_aug, pay_amt_jul, pay_amt_jun, pay_amt_may, pay_amt_apr,
        utilisation_ratio, payment_trend, missed_payments
    ]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    risk_prob = model.predict_proba(input_scaled)[0][1]
    
    # Display result
    st.subheader("Risk Assessment")
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric("Default Probability", f"{risk_prob:.1%}")
    
    with col_res2:
        st.metric("Missed Payments", f"{missed_payments}/6")
    
    with col_res3:
        st.metric("Utilisation Ratio", f"{utilisation_ratio:.2f}")
    
    # Decision
    st.subheader("Lending Decision")
    if risk_prob < 0.25:
        st.success(f"✅ AUTO-APPROVE — Risk score {risk_prob:.2f} is below threshold (0.25)")
        st.write("Low risk customer. Recommend approval at standard terms.")
    elif risk_prob > 0.45:
        st.error(f"❌ AUTO-DECLINE — Risk score {risk_prob:.2f} exceeds upper threshold (0.45)")
        st.write("High risk customer. Recommend decline.")
    else:
        st.warning(f"⚠️ HUMAN REVIEW — Risk score {risk_prob:.2f} falls in review zone (0.25–0.45)")
        st.write("Borderline case. Recommend manual underwriter review.")
    
    # Business context
    with st.expander("📊 Business Impact Context"):
        st.write(f"""
        **Model Performance:** AUC-ROC = 0.771  
        **Optimal Threshold:** 0.25  
        **Expected Saving:** £12M per 6,000 applications (61% cost reduction vs baseline)  
        **Decision Logic:**
        - Auto-approve: risk < 0.25 (low risk, fast decision)
        - Human review: risk 0.25–0.45 (borderline, underwriter judgement)
        - Auto-decline: risk > 0.45 (high risk, protect portfolio)
        """)

st.caption("Built by Bader Matar | MSc AI, University of Sheffield | Credit Default Dataset (UCI)")