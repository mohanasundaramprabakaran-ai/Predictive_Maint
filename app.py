
import streamlit as st
import pandas as pd
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "xgboost_model.pkl"
)

DEFAULTS_PATH = os.path.join(
    PROJECT_DIR,
    "feature_defaults.json"
)

FEATURE_COLUMNS = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temperature",
    "Coolant_Temperature"
]


# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Load default sensor values
@st.cache_data
def load_defaults():
    with open(DEFAULTS_PATH, "r") as file:
        return json.load(file)


# Application title
st.title("⚙️ Engine Predictive Maintenance")
st.subheader("Engine Condition Prediction System")

st.write(
    "Enter the engine sensor readings to predict "
    "the engine condition using the XGBoost model."
)


# Validate deployment files
if not os.path.exists(MODEL_PATH):
    st.error("Trained model file is missing.")
    st.stop()

if not os.path.exists(DEFAULTS_PATH):
    st.error("Feature defaults file is missing.")
    st.stop()


# Load model and defaults
try:
    model = load_model()
    defaults = load_defaults()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


st.divider()

st.subheader("Engine Sensor Inputs")

col1, col2 = st.columns(2)

with col1:

    engine_rpm = st.number_input(
        "Engine RPM",
        value=float(defaults["Engine_RPM"]),
        format="%.2f"
    )

    lub_oil_pressure = st.number_input(
        "Lub Oil Pressure",
        value=float(defaults["Lub_Oil_Pressure"]),
        format="%.4f"
    )

    fuel_pressure = st.number_input(
        "Fuel Pressure",
        value=float(defaults["Fuel_Pressure"]),
        format="%.4f"
    )


with col2:

    coolant_pressure = st.number_input(
        "Coolant Pressure",
        value=float(defaults["Coolant_Pressure"]),
        format="%.4f"
    )

    lub_oil_temperature = st.number_input(
        "Lub Oil Temperature",
        value=float(defaults["Lub_Oil_Temperature"]),
        format="%.4f"
    )

    coolant_temperature = st.number_input(
        "Coolant Temperature",
        value=float(defaults["Coolant_Temperature"]),
        format="%.4f"
    )


st.divider()


# Prediction button
if st.button(
    "Predict Engine Condition",
    type="primary"
):

    input_data = pd.DataFrame([{
        "Engine_RPM": engine_rpm,
        "Lub_Oil_Pressure": lub_oil_pressure,
        "Fuel_Pressure": fuel_pressure,
        "Coolant_Pressure": coolant_pressure,
        "Lub_Oil_Temperature": lub_oil_temperature,
        "Coolant_Temperature": coolant_temperature
    }], columns=FEATURE_COLUMNS)

    try:

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)

        predicted_index = classes.index(prediction)

        confidence = float(probabilities[predicted_index])

        st.subheader("Prediction Result")

        st.metric(
            "Predicted Engine Condition",
            f"Class {prediction}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )

        probability_df = pd.DataFrame({
            "Engine Condition": [
                f"Class {c}" for c in classes
            ],
            "Probability (%)": [
                round(float(p) * 100, 2)
                for p in probabilities
            ]
        })

        st.write("Prediction Probabilities")

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Class 0 and Class 1 represent the target labels "
            "in the training dataset. Confirm their meanings "
            "before interpreting them as healthy or faulty."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


st.divider()

st.caption(
    "Powered by XGBoost | Predictive Maintenance Project"
)
