import streamlit as st
import pandas as pd
import numpy as np
import pickle

LOW_THRESHOLD = 0.5086
HIGH_THRESHOLD = 0.5594

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_data
def load_reference_data():
    claims = pd.read_csv("/Users/priyalshah/Documents/insurance-claims-ai/data/claims_features.csv")
    return claims

@st.cache_resource
def load_model():
    with open("/Users/priyalshah/Documents/insurance-claims-ai/modeling/denial_model.pkl", "rb") as f:
        return pickle.load(f)

claims_df = load_reference_data()
model = load_model()

# Risk thresholds (business-aligned)
LOW_RISK = claims_df["is_denied"].mean()
MED_RISK = claims_df["is_denied"].quantile(0.60)
HIGH_RISK = claims_df["is_denied"].quantile(0.85)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Claim Decision Support", layout="centered")

st.title("🏥 Insurance Claim Decision Support Tool")
st.caption("Pre-submission guidance for hospital billing teams")

st.divider()

# ----------------------------
# User Inputs
# ----------------------------
st.subheader("Claim Details")

provider_specialty = st.selectbox(
    "Provider Specialty",
    claims_df["provider_specialty"].unique()
)

cpt_code = st.selectbox(
    "CPT Code",
    claims_df["cpt_code"].unique()
)

charge_amount = st.number_input(
    "Charge Amount ($)",
    min_value=50.0,
    step=10.0
)

prior_auth = st.radio(
    "Prior Authorization Obtained?",
    ["Yes", "No"]
)

modifier = st.selectbox(
    "Modifier",
    ["None", "26", "TC"]
)

submit = st.button("Evaluate Claim")

# ----------------------------
# Decision Logic
# ----------------------------
if submit:
    st.divider()
    st.subheader("Decision")

    # ---- Hard rule checks ----
    hard_fail_reasons = []

    requires_auth = claims_df[
        claims_df["cpt_code"] == cpt_code
    ]["failed_prior_auth"].mean() > 0

    if requires_auth and prior_auth == "No":
        hard_fail_reasons.append("Missing required prior authorization")

    if modifier == "None" and cpt_code in ["93000", "70551"]:
        hard_fail_reasons.append("Modifier required for this CPT code")

    # ---- Outcome if hard rule fails ----
    if hard_fail_reasons:
        st.error("🔴 High Risk: Likely Denial")
        for r in hard_fail_reasons:
            st.write(f"- {r}")
        st.info("Recommendation: Fix the issues above before submitting.")
    
    else:
        # ---- ML risk scoring ----
        provider_rate = claims_df[
            claims_df["provider_specialty"] == provider_specialty
        ]["provider_denial_rate"].mean()

        avg_charge = claims_df[
            claims_df["cpt_code"] == cpt_code
        ]["avg_cpt_charge"].mean()

        charge_ratio = charge_amount / avg_charge

        X = pd.DataFrame([{
            "provider_denial_rate": provider_rate,
            "provider_claim_volume": 50,
            "charge_to_avg_ratio": charge_ratio,
            "high_charge_flag": charge_ratio > 1.5,
            "failed_prior_auth": False,
            "days_since_submission": 0
        }])

        risk = model.predict_proba(X)[0][1]

        # ---- Decision buckets ----
        if risk >= HIGH_THRESHOLD:
            st.error("🔴 High Risk – Likely Denial")
            st.write(f"Predicted denial risk: {risk:.1%}")
            st.info(
                "Recommendation: High-risk claim. Review authorization status, "
                "charge level, and coding before submission."
            )

        elif risk >= LOW_THRESHOLD:
            st.warning("🟡 Review Recommended")
            st.write(f"Predicted denial risk: {risk:.1%}")
            st.info(
                "Recommendation: Moderate risk. A quick review may prevent denial."
            )

        else:
            st.success("🟢 Safe to Submit")
            st.write(f"Predicted denial risk: {risk:.1%}")
            st.info(
                "Recommendation: Low predicted denial risk. Claim can be submitted as-is."
            )

