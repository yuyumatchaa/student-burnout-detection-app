import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


st.set_page_config(
    page_title="Student Mental Health and Burnout Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Student Mental Health and Burnout Detection")
st.write("This application predicts whether a student may be at risk of depression or burnout.")


@st.cache_resource
def train_model():
    df = pd.read_csv("student_depression_dataset.csv")

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    target_col = "Depression"

    app_features = [
        "Age",
        "Academic Pressure",
        "CGPA",
        "Study Satisfaction",
        "Sleep Duration",
        "Dietary Habits",
        "Work/Study Hours",
        "Financial Stress",
        "Family History of Mental Illness"
    ]

    df = df[app_features + [target_col]].copy()

    # Use sample to make cloud startup faster
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

    X = df[app_features]
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

    category_options = {}
    for col in categorical_features:
        category_options[col] = sorted(df[col].dropna().unique().tolist())

    return pipeline, metrics, category_options


with st.spinner("Training model... Please wait."):
    model, metrics, category_options = train_model()

st.success("Model loaded successfully!")

st.subheader("Enter Student Information")

age = st.number_input("Age", min_value=15, max_value=60, value=22)
academic_pressure = st.slider("Academic Pressure", 0, 5, 3)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3)

sleep_duration = st.selectbox(
    "Sleep Duration",
    category_options.get("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
)

dietary_habits = st.selectbox(
    "Dietary Habits",
    ["Healthy", "Moderate", "Unhealthy"]
)

work_study_hours = st.slider("Work/Study Hours", 0, 12, 6)
financial_stress = st.slider("Financial Stress", 1, 5, 3)

family_history = st.selectbox(
    "Family History of Mental Illness",
    category_options.get("Family History of Mental Illness", ["No", "Yes"])
)

input_data = pd.DataFrame([{
    "Age": age,
    "Academic Pressure": academic_pressure,
    "CGPA": cgpa,
    "Study Satisfaction": study_satisfaction,
    "Sleep Duration": sleep_duration,
    "Dietary Habits": dietary_habits,
    "Work/Study Hours": work_study_hours,
    "Financial Stress": financial_stress,
    "Family History of Mental Illness": family_history
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
    else:
        probability = 0

    if prediction == 1:
        st.error(f"Prediction: High Risk")
        st.write(f"Estimated risk probability: {probability:.2%}")
    else:
        st.success(f"Prediction: Low Risk")
        st.write(f"Estimated risk probability: {probability:.2%}")

st.subheader("Model Performance")
st.write(pd.DataFrame([metrics]))

st.info("This application is an early screening tool only and does not replace professional medical diagnosis.")
