import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="Student Burnout Detection",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("student_depression_dataset.csv")
    df.columns = df.columns.str.strip()

    target_col = "Depression"

    # Remove id column if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Select features for web app
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

    app_features = [col for col in app_features if col in df.columns]

    X = df[app_features]
    y = df[target_col]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Preprocessing
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot)
            ]), categorical_cols)
        ]
    )

    # Complete ML pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=1
        ))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    evaluation = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    # Category options for dropdown
    category_options = {}
    for col in app_features:
        if df[col].dtype == "object":
            category_options[col] = sorted(df[col].dropna().astype(str).unique().tolist())

    return model, app_features, category_options, evaluation


st.title("🎓 Smart Campus Student Mental Health and Burnout Detection")

st.write("""
This web application uses a Machine Learning model to predict whether a student may be at risk of depression or burnout.
The prediction is based on academic, lifestyle, and stress-related factors.
""")

st.warning("""
Disclaimer: This system is only an early screening tool. It is not a medical diagnosis.
Students are encouraged to contact a counsellor or healthcare professional for proper support.
""")

# Train/load model
try:
    model, app_features, category_options, evaluation = train_model()
except FileNotFoundError:
    st.error("Dataset file not found. Please make sure 'student_depression_dataset.csv' is in the same folder as app.py.")
    st.stop()

st.success("Machine Learning model loaded successfully.")

with st.expander("Show Model Evaluation Result"):
    st.write("Accuracy:", round(evaluation["Accuracy"], 4))
    st.write("Precision:", round(evaluation["Precision"], 4))
    st.write("Recall:", round(evaluation["Recall"], 4))
    st.write("F1-score:", round(evaluation["F1-score"], 4))
    st.write("Confusion Matrix:")
    st.write(evaluation["Confusion Matrix"])

st.subheader("Enter Student Information")

user_input = {}

for feature in app_features:

    if feature == "Age":
        user_input[feature] = st.number_input("Age", min_value=10, max_value=80, value=22)

    elif feature == "Academic Pressure":
        user_input[feature] = st.slider("Academic Pressure", min_value=0, max_value=5, value=3)

    elif feature == "CGPA":
        user_input[feature] = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)

    elif feature == "Study Satisfaction":
        user_input[feature] = st.slider("Study Satisfaction", min_value=0, max_value=5, value=3)

    elif feature == "Work/Study Hours":
        user_input[feature] = st.slider("Work/Study Hours per Day", min_value=0, max_value=16, value=6)

    elif feature == "Financial Stress":
        user_input[feature] = st.slider("Financial Stress", min_value=0, max_value=5, value=3)

    elif feature in category_options:
        user_input[feature] = st.selectbox(feature, category_options[feature])

    else:
        user_input[feature] = st.text_input(feature)

input_df = pd.DataFrame([user_input])
input_df = input_df[app_features]

st.subheader("Input Preview")
st.dataframe(input_df)

if st.button("Predict Risk"):

    prediction = model.predict(input_df)[0]

    try:
        probability = model.predict_proba(input_df)[0][1]
    except:
        probability = None

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk: The student may show signs of depression or burnout risk.")
        st.write("""
        Recommendation: The student is encouraged to contact the campus counselling unit,
        student affairs department, or a qualified mental health professional for early support.
        """)
    else:
        st.success("Low Risk: The student is not predicted as high risk based on the provided information.")
        st.write("""
        Recommendation: The student should continue maintaining healthy study habits,
        good sleep, and a balanced lifestyle.
        """)

    if probability is not None:
        st.write(f"Estimated risk probability: **{probability:.2%}**")

st.markdown("---")
st.caption("BCI3333 Machine Learning Applications - Smart Campus Project")