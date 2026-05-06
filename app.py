"""
Heart Disease Risk Estimator
Streamlit app wrapping a random forest trained on the Cleveland Heart Disease dataset.

This is a research and educational demo. It is not a clinical tool and must
not be used for medical decision-making.
"""

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Page config and disclaimer
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Risk Estimator",
    page_icon="❤️",
    layout="wide",
)

st.title("Heart Disease Risk Estimator")
st.markdown(
    "A demo of a random forest classifier trained on the "
    "[Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) "
    "(303 patients, 13 clinical features). Enter values on the left, then click "
    "**Estimate risk** to see the model's prediction and which features drove it."
)

st.warning(
    "**Research demo only.** This tool is not validated for clinical use, "
    "is trained on a small public research dataset, and must not be used "
    "to diagnose, screen, or make treatment decisions for any individual."
)

# -----------------------------------------------------------------------------
# Load model and SHAP explainer (cached so it runs once per session)
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("heart_rf_model.joblib")
    feature_names = joblib.load("feature_names.joblib")
    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer


model, feature_names, explainer = load_model()

# -----------------------------------------------------------------------------
# Input form
# -----------------------------------------------------------------------------

# Friendly labels and option mappings for the categorical features.
SEX_OPTIONS = {"Female": 0, "Male": 1}
CP_OPTIONS = {
    "Typical angina": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "Asymptomatic": 3,
}
FBS_OPTIONS = {"≤ 120 mg/dl": 0, "> 120 mg/dl": 1}
RESTECG_OPTIONS = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}
EXANG_OPTIONS = {"No": 0, "Yes": 1}
SLOPE_OPTIONS = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL_OPTIONS = {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}

st.sidebar.header("Patient features")
st.sidebar.caption(
    "Defaults reflect a representative middle-aged patient. Adjust to explore "
    "how predictions change."
)

age = st.sidebar.slider("Age", 29, 77, 54, help="Age in years.")
sex_label = st.sidebar.selectbox("Sex", list(SEX_OPTIONS.keys()), index=1)
cp_label = st.sidebar.selectbox(
    "Chest pain type",
    list(CP_OPTIONS.keys()),
    index=3,
    help="Type of chest pain experienced.",
)
trestbps = st.sidebar.slider(
    "Resting blood pressure (mm Hg)", 94, 200, 130,
    help="Resting systolic blood pressure on admission.",
)
chol = st.sidebar.slider(
    "Serum cholesterol (mg/dl)", 126, 564, 245,
)
fbs_label = st.sidebar.selectbox(
    "Fasting blood sugar",
    list(FBS_OPTIONS.keys()),
    index=0,
    help="Whether fasting blood sugar exceeds 120 mg/dl.",
)
restecg_label = st.sidebar.selectbox(
    "Resting ECG result", list(RESTECG_OPTIONS.keys()), index=0,
)
thalach = st.sidebar.slider(
    "Maximum heart rate achieved", 71, 202, 150,
    help="Peak heart rate during exercise stress testing.",
)
exang_label = st.sidebar.selectbox(
    "Exercise-induced angina", list(EXANG_OPTIONS.keys()), index=0,
)
oldpeak = st.sidebar.slider(
    "ST depression (oldpeak)", 0.0, 6.2, 1.0, 0.1,
    help="ST depression induced by exercise relative to rest.",
)
slope_label = st.sidebar.selectbox(
    "Slope of peak exercise ST segment", list(SLOPE_OPTIONS.keys()), index=1,
)
ca = st.sidebar.slider(
    "Number of major vessels colored by fluoroscopy", 0, 3, 0,
)
thal_label = st.sidebar.selectbox(
    "Thalassemia", list(THAL_OPTIONS.keys()), index=0,
)

predict_clicked = st.sidebar.button("Estimate risk", type="primary")

# -----------------------------------------------------------------------------
# Build the feature row in the exact order the model expects
# -----------------------------------------------------------------------------

input_row = pd.DataFrame(
    [[
        age,
        SEX_OPTIONS[sex_label],
        CP_OPTIONS[cp_label],
        trestbps,
        chol,
        FBS_OPTIONS[fbs_label],
        RESTECG_OPTIONS[restecg_label],
        thalach,
        EXANG_OPTIONS[exang_label],
        oldpeak,
        SLOPE_OPTIONS[slope_label],
        ca,
        THAL_OPTIONS[thal_label],
    ]],
    columns=feature_names,
)

# -----------------------------------------------------------------------------
# Output: prediction, probability gauge, SHAP explanation
# -----------------------------------------------------------------------------

if predict_clicked:
    proba = model.predict_proba(input_row)[0, 1]
    pred = int(proba >= 0.5)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Prediction")
        if pred == 1:
            st.error(f"**Heart disease predicted**\n\nProbability: {proba:.1%}")
        else:
            st.success(f"**No heart disease predicted**\n\nProbability: {proba:.1%}")

        st.markdown("**Risk probability**")
        st.progress(float(proba))
        st.caption(
            "Threshold for predicting disease is 0.5. In a real clinical "
            "deployment this threshold would be calibrated to the cost of "
            "false positives versus false negatives."
        )

    with col2:
        st.subheader("Why this prediction")
        st.caption(
            "Top features driving this individual prediction, computed with "
            "SHAP values. Bars to the right push the prediction toward heart "
            "disease; bars to the left push it away."
        )

        # SHAP for a single row
        shap_values = explainer.shap_values(input_row)

        # TreeExplainer returns either an array of shape (1, n_features) for the
        # positive class or a list/3D array containing both classes depending on
        # sklearn and shap versions. Normalize to a 1-D vector for class 1.
        if isinstance(shap_values, list):
            class1_shap = shap_values[1][0]
        else:
            arr = np.array(shap_values)
            if arr.ndim == 3:
                class1_shap = arr[0, :, 1]
            else:
                class1_shap = arr[0]

        contrib = pd.DataFrame({
            "feature": feature_names,
            "value": input_row.iloc[0].values,
            "shap": class1_shap,
        })
        contrib["abs_shap"] = contrib["shap"].abs()
        contrib = contrib.sort_values("abs_shap", ascending=True).tail(5)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in contrib["shap"]]
        ax.barh(contrib["feature"], contrib["shap"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on heart disease probability)")
        ax.set_title("Top 5 contributing features")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Enter patient features in the sidebar and click **Estimate risk**.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Model: Random Forest Classifier. Test-set accuracy 0.869, ROC AUC 0.93. "
    "Source code on [GitHub](https://github.com/Nelsike). "
    "Built by Joshua Sanchez."
)
