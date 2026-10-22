import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Titanic Passenger Survival Prediction
# Keep app.py and svc_trained.pkl in the same folder/repository.
# IMPORTANT: Input columns and encoding must match your notebook.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "svc_trained.pkl"

st.title("🚢 Titanic Passenger Survival Prediction")
st.write(
    "Enter passenger information below to estimate whether the passenger "
    "would have survived the Titanic disaster."
)

@st.cache_resource
def load_model(path):
    with open(path, "rb") as model_file:
        return pickle.load(model_file)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Upload `svc_trained.pkl` to the same GitHub "
        "repository folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.subheader("Passenger details")

with st.form("passenger_form"):
    pclass = st.selectbox("Passenger class", options=[1, 2, 3], index=2)
    sex = st.selectbox("Sex", options=["Female", "Male"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
    sibsp = st.number_input(
        "Siblings / spouses aboard", min_value=0, max_value=10, value=0, step=1
    )
    parch = st.number_input(
        "Parents / children aboard", min_value=0, max_value=10, value=0, step=1
    )
    fare = st.number_input(
        "Ticket fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0
    )
    submitted = st.form_submit_button("Predict survival", type="primary")

if submitted:
    # Common encoding used by some Titanic teaching notebooks.
    # Change this if your notebook uses a different LabelEncoder mapping.
    sex_encoded = 0 if sex == "Female" else 1

    # The feature names/order must match the model's training data.
    input_data = pd.DataFrame(
        [[pclass, sex_encoded, age, sibsp, parch, fare]],
        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],
    )

    try:
        prediction = model.predict(input_data)[0]

        # Common Titanic labels: 0 = did not survive, 1 = survived.
        # If your notebook uses reversed or text labels, adjust this section.
        if str(prediction).strip() in {"1", "1.0", "Survived", "survived"}:
            st.success("Prediction: The passenger is predicted to have survived.")
        elif str(prediction).strip() in {
            "0", "0.0", "Did not survive", "Not Survived", "not survived"
        }:
            st.error("Prediction: The passenger is predicted not to have survived.")
        else:
            st.info(f"Model prediction: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.caption(
                "Prediction probabilities (class order: "
                + ", ".join(map(str, model.classes_))
                + ")"
            )
            st.write(
                {
                    str(label): round(float(probability), 4)
                    for label, probability in zip(model.classes_, probabilities)
                }
            )
    except Exception as exc:
        st.error(
            "Prediction failed. Your model may expect different feature columns, "
            "column order, or encoded inputs. Check the training notebook and "
            f"update the input section in app.py. Technical details: {exc}"
        )

st.divider()
st.caption(
    "Educational demonstration only. This model predicts from learned patterns "
    "and cannot establish what would have happened to a real person."
)
