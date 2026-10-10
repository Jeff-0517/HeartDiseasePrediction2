import os
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="HeartGuard | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
def find_model_path():
    candidates = [
        "heart_disease_model.joblib",
        "heart_disease_model (1).joblib",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No model file was found in the app folder. "
        "Please keep heart_disease_model.joblib or heart_disease_model (1).joblib."
    )


@st.cache_resource
def load_model():
    return joblib.load(find_model_path())


try:
    model = load_model()
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.write(
        "This usually means the model was trained with a different scikit-learn version. "
        "Use Python 3.13 and install scikit-learn==1.6.1."
    )
    st.stop()

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 750;
        color: #176B87;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 16px;
        color: #64748b;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        color: #176B87;
    }

    div.stButton > button {
        width: 100%;
        background-color: #176B87;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #12566C;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="main-title">❤️ HeartGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based Heart Disease Risk Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Academic prototype only. Predictions are not medical diagnoses "
    "and must not be used to make treatment decisions."
)

# -----------------------------
# PROJECT SUMMARY DASHBOARD
# -----------------------------
st.subheader("Model Performance at a Glance")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Test Accuracy", "89.67%")

with metric2:
    st.metric("Positive-Class Recall", "92.16%")

with metric3:
    st.metric("F1-Score", "90.82%")

with metric4:
    st.metric("Test Records", "184")

st.caption(
    "Metrics are from the held-out test dataset used during model evaluation."
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("About HeartGuard")

    st.write(
        "HeartGuard uses a trained Random Forest machine learning "
        "model to classify clinical input data."
    )

    st.divider()

    st.subheader("Model Information")
    st.write("Algorithm: Random Forest")
    st.write("Training dataset: Kaggle Heart Failure Prediction")
    st.write("Held-out test accuracy: 89.67%")

    st.caption(
        "Accuracy is a dataset-level evaluation result, "
        "not an individual's probability of being correctly diagnosed."
    )

    st.divider()

    st.caption("Final-year academic project")

# -----------------------------
# PATIENT INPUT FORM
# -----------------------------
st.markdown(
    '<div class="section-title">Patient Clinical Information</div>',
    unsafe_allow_html=True
)

st.write("Enter the available clinical measurements below.")

with st.form("patient_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Basic Information")

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=40
        )

        sex = st.selectbox("Sex", ["M", "F"])

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"],
            format_func=lambda x: {
                "ATA": "ATA - Atypical Angina",
                "NAP": "NAP - Non-Anginal Pain",
                "ASY": "ASY - Asymptomatic",
                "TA": "TA - Typical Angina"
            }[x]
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["N", "Y"],
            format_func=lambda x: "No" if x == "N" else "Yes"
        )

    with col2:
        st.subheader("Clinical Measurements")

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=1,
            max_value=300,
            value=120
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=1,
            max_value=700,
            value=200
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL?",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=1,
            max_value=250,
            value=150
        )

    with col3:
        st.subheader("Additional Results")

        resting_ecg = st.selectbox(
            "Resting ECG Result",
            ["Normal", "ST", "LVH"]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-5.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

        st.write("")
        st.caption("Check that all entries reflect the intended test data.")

    submitted = st.form_submit_button(
        "Generate Prediction"
    )

# -----------------------------
# PREDICTION
# -----------------------------
if submitted:

    if resting_bp == 0 or cholesterol == 0:
        st.error(
            "Blood pressure and cholesterol must be greater than zero."
        )
        st.stop()

    patient_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    try:
        if hasattr(model, "feature_names_in_"):
            patient_data = patient_data[model.feature_names_in_]

        prediction = model.predict(patient_data)[0]
        probabilities = model.predict_proba(patient_data)[0]

        class_index = list(model.classes_).index(1)
        positive_probability = probabilities[class_index]

        st.session_state.prediction_history.append({
            "Age": age,
            "Sex": sex,
            "Prediction": (
                "Positive (1)" if prediction == 1 else "Negative (0)"
            ),
            "Positive-Class Probability": (
                f"{positive_probability * 100:.2f}%"
            )
        })

        st.divider()
        st.header("Prediction Dashboard")

        result_col, probability_col = st.columns(2)

        with result_col:
            st.subheader("Model Classification")

            if prediction == 1:
                st.error(
                    "Positive class predicted "
                    "(HeartDisease = 1)"
                )
            else:
                st.success(
                    "Negative class predicted "
                    "(HeartDisease = 0)"
                )

        with probability_col:
            st.subheader("Model Probability")

            st.metric(
                "Positive-class probability",
                f"{positive_probability * 100:.2f}%"
            )

            st.progress(float(positive_probability))

        st.caption(
            "This probability is generated by the model. "
            "It is not a clinically validated estimate of an individual's "
            "actual risk of heart disease."
        )

        # Generate a downloadable prediction report
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prediction_label = (
            "Positive class (HeartDisease = 1)"
            if prediction == 1
            else "Negative class (HeartDisease = 0)"
        )

        report_text = f"""
HEARTGUARD
HEART DISEASE PREDICTION REPORT
================================

Report generated: {report_time}

PATIENT CLINICAL INFORMATION
----------------------------
Age: {age}
Sex: {sex}
Chest Pain Type: {chest_pain}
Resting Blood Pressure: {resting_bp} mm Hg
Cholesterol: {cholesterol} mg/dL
Fasting Blood Sugar: {fasting_bs}
Resting ECG: {resting_ecg}
Maximum Heart Rate: {max_hr}
Exercise-Induced Angina: {exercise_angina}
Oldpeak: {oldpeak}
ST Slope: {st_slope}

MODEL RESULT
------------
Predicted Class: {prediction_label}
Model-estimated positive-class probability: {positive_probability * 100:.2f}%

IMPORTANT NOTICE
----------------
This report was generated by an academic machine learning prototype.
It is not a medical diagnosis and must not be used to make treatment
decisions. Consult a qualified healthcare professional for medical advice.
"""

        st.download_button(
            label="Download Prediction Report (.txt)",
            data=report_text,
            file_name="heartguard_prediction_report.txt",
            mime="text/plain"
        )

        with st.expander("View submitted clinical data"):
            st.dataframe(patient_data, width="stretch")

    except Exception as error:
        st.error("An error occurred while generating the prediction.")
        st.code(str(error))

st.divider()
st.header("Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, width="stretch")

    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction History (CSV)",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("No predictions have been generated in this session yet.")

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.divider()
st.header("Model Performance Evaluation")
st.write(
    "The following results were obtained during evaluation "
    "on the held-out test dataset."
)

performance_data = pd.DataFrame({
    "Algorithm": [
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "Gradient Boosting"
    ],
    "Accuracy": [89.67, 88.59, 87.50, 86.96],
    "Precision": [89.52, 88.57, 88.35, 89.00],
    "Recall": [92.16, 91.18, 89.22, 87.25],
    "F1-Score": [90.82, 89.86, 88.78, 88.12],
    "ROC-AUC": [93.19, 93.29, 93.99, 91.44]
})

st.dataframe(
    performance_data.style.format({
        "Accuracy": "{:.2f}%",
        "Precision": "{:.2f}%",
        "Recall": "{:.2f}%",
        "F1-Score": "{:.2f}%",
        "ROC-AUC": "{:.2f}%"
    }),
    width="stretch",
    hide_index=True
)

st.subheader("Accuracy Comparison")
chart_data = performance_data.set_index("Algorithm")[["Accuracy"]]
st.bar_chart(chart_data)

st.caption(
    "These are held-out test-set results, not clinical validation. "
    "The percentages are reproduced from the model evaluation "
    "results obtained during development."
)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
st.divider()
st.header("Confusion Matrix")

st.write(
    "The confusion matrix shows the Random Forest model's "
    "predictions compared with the actual test-set labels."
)

try:
    cm = [[71, 11], [8, 94]]

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No Disease", "Predicted Disease"],
        yticklabels=["Actual No Disease", "Actual Disease"],
        ax=ax
    )

    ax.set_xlabel("Model Prediction")
    ax.set_ylabel("Actual Class")
    ax.set_title("Random Forest Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Interpretation**
    - **71 true negatives:** correctly predicted the negative class.
    - **11 false positives:** predicted positive when the actual class was negative.
    - **8 false negatives:** predicted negative when the actual class was positive.
    - **94 true positives:** correctly predicted the positive class.
    """)

    st.caption(
        "These counts are from the held-out test set of 184 records. "
        "They are evaluation results, not evidence of clinical validity."
    )

except Exception as error:
    st.error(f"Could not display the confusion matrix: {error}")

# -----------------------------
# SYSTEM TESTING
# -----------------------------
st.divider()
st.header("System Testing")

st.write(
    "Run a basic functional test to verify that the trained model "
    "accepts clinical inputs and returns a valid prediction."
)

if st.button("Run System Test"):

    test_patient = pd.DataFrame([{
        "Age": 55,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 250,
        "FastingBS": 1,
        "RestingECG": "ST",
        "MaxHR": 130,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.0,
        "ST_Slope": "Flat"
    }])

    try:
        test_prediction = model.predict(test_patient)[0]
        test_probabilities = model.predict_proba(test_patient)[0]

        valid_prediction = test_prediction in model.classes_
        valid_probabilities = (
            len(test_probabilities) == len(model.classes_)
            and all(0 <= p <= 1 for p in test_probabilities)
            and abs(sum(test_probabilities) - 1.0) < 1e-6
        )

        st.write("**Test Results**")

        if valid_prediction:
            st.success("PASS: Model generated a valid class prediction.")
        else:
            st.error("FAIL: Model returned an unexpected class.")

        if valid_probabilities:
            st.success("PASS: Probability values are valid and sum to 1.")
        else:
            st.error("FAIL: Probability validation failed.")

        if valid_prediction and valid_probabilities:
            st.success("Basic functional tests passed.")
        else:
            st.warning("One or more tests failed. Review the results.")

    except Exception as error:
        st.error("System test failed.")
        st.code(str(error))

st.caption(
    "This checks basic software functionality only. "
    "It does not establish medical accuracy or clinical safety."
)

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "HeartGuard | Machine Learning Academic Project | "
    "For educational and research demonstration only."
)
