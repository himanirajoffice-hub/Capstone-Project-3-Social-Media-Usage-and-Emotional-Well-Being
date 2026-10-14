from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Social Media and Emotional Well-Being", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "emotion_classifier.joblib")


st.title("Social Media and Emotional Well-Being")
st.caption("Capstone Project 3 | Himani Verma")
st.write("Explore the emotion label predicted by a Random Forest model from social media activity.")
st.info("Educational demonstration. Predictions are dataset labels, not a mental health assessment. Model probabilities are not calibrated measures of certainty.")

try:
    model = load_model()
except Exception as exc:
    st.error("The model could not be loaded. Check that emotion_classifier.joblib is beside app.py and install the versions in requirements.txt.")
    st.exception(exc)
    st.stop()

predict_tab, results_tab, about_tab = st.tabs(["Predict an emotion", "Model results", "About the project"])

with predict_tab:
    st.subheader("Enter a social media activity profile")
    st.caption("The inputs are limited to the ranges observed in the training data. Gender can be left unknown.")
    with st.form("activity_profile"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", min_value=21, max_value=35, value=25, step=1)
            gender = st.selectbox("Gender", ["Female", "Male", "Non-binary", "Unknown"])
            platform = st.selectbox("Platform", ["Instagram", "Twitter", "Facebook", "LinkedIn", "Whatsapp", "Telegram", "Snapchat"])
            usage = st.number_input("Daily usage time in minutes", min_value=30, max_value=200, value=120, step=1)
        with right:
            posts = st.number_input("Posts per day", min_value=1, max_value=10, value=3, step=1)
            likes = st.number_input("Likes received per day", min_value=5, max_value=110, value=45, step=1)
            comments = st.number_input("Comments received per day", min_value=2, max_value=40, value=10, step=1)
            messages = st.number_input("Messages sent per day", min_value=8, max_value=50, value=12, step=1)
        submitted = st.form_submit_button("Predict emotion", type="primary")

    if submitted:
        profile = pd.DataFrame([{
            "Age": age,
            "Gender": np.nan if gender == "Unknown" else gender,
            "Platform": platform,
            "Daily_Usage_Time (minutes)": usage,
            "Posts_Per_Day": posts,
            "Likes_Received_Per_Day": likes,
            "Comments_Received_Per_Day": comments,
            "Messages_Sent_Per_Day": messages,
        }])
        profile = profile[list(model.feature_names_in_)]
        try:
            prediction = model.predict(profile)[0]
            probabilities = model.predict_proba(profile)[0]
            st.subheader(f"Predicted emotion: {prediction}")
            probability_df = pd.DataFrame({"Emotion": model.classes_, "Model probability": probabilities}).sort_values("Model probability", ascending=False)
            st.bar_chart(probability_df.set_index("Emotion"))
            st.dataframe(probability_df, hide_index=True, column_config={"Model probability": st.column_config.NumberColumn(format="%.3f")})
        except Exception as exc:
            st.error("Prediction could not be completed.")
            st.exception(exc)

with results_tab:
    metrics = pd.read_csv(ROOT / "final_test_metrics.csv").iloc[0]
    a, b, c = st.columns(3)
    a.metric("Test accuracy", f"{metrics['Test_Accuracy']:.2%}")
    b.metric("Test Macro F1", f"{metrics['Test_Macro_F1']:.3f}")
    c.metric("Test records", int(metrics["Test_Records"]))
    st.caption("These are saved held-out test results from the notebook, not scores for the profile entered above.")
    st.subheader("Validation model comparison")
    st.dataframe(pd.read_csv(ROOT / "model_comparison.csv").round(3), hide_index=True)
    st.write("Random Forest was selected using validation Macro F1. Tuning selected the original settings and did not improve validation performance.")
    st.subheader("Held-out test predictions")
    predictions = pd.read_csv(ROOT / "test_predictions.csv")
    st.dataframe(predictions, hide_index=True)

with about_tab:
    st.subheader("Objective and method")
    st.write("Predict one of six dominant emotion labels using age, gender, platform and five daily activity measures. The project compares a majority-class baseline, Logistic Regression, Decision Tree and Random Forest.")
    st.write("The source files contained formatting errors, swapped Age/Gender values and overlapping records. After cleaning, excluding one uncertain target label, removing User_ID and deduplicating, 265 records were split into 159 training, 53 validation and 53 test records.")
    st.write("Preprocessing includes missing-value handling, numerical standardization and one-hot encoding. Tuning used five-fold cross-validation within training data. The selected pipeline was refitted on 212 development records before final test evaluation.")
    st.subheader("Limits")
    st.write("The small dataset, data-quality problems and training-validation gap limit generalization. Higher usage and an emotion occurring together do not establish causation. No user profiles are saved by this app.")
    st.markdown("[Dataset source on Kaggle](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being/)")
