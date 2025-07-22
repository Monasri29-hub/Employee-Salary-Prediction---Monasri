import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="ML Classifier App", layout="wide")
st.title("📊 AI Classifier from Uploaded File")

# Upload file section
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load the file based on its extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("1️⃣ Uploaded Data Preview")
    st.dataframe(df.head())

    # Automatically detect and display categorical columns
    st.subheader("2️⃣ Label Encoding for Categorical Columns")
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']):
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        except Exception as e:
            st.warning(f"Could not encode column `{col}`: {e}")

    # Automatically detect target column (last column)
    target_column = df.columns[-1]
    st.success(f"✅ Auto-selected Target Column: `{target_column}`")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("3️⃣ Accuracy & Classification Report")
    st.metric("Model Accuracy", f"{acc:.2%}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred), language='text')

    # Optional plotting
    st.subheader("4️⃣ Feature Importance")
    importances = clf.feature_importances_
    features = X.columns
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(features, importances, color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')
    st.pyplot(fig)

    # Try Prediction on new data
    st.subheader("5️⃣ Predict with Custom Input")
    input_data = {}
    for col in X.columns:
        value = st.number_input(f"Enter value for `{col}`", value=float(X[col].mean()))
        input_data[col] = value

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = clf.predict(input_df)[0]

        if target_column in label_encoders:
            pred_label = label_encoders[target_column].inverse_transform([pred])[0]
            st.success(f"🎯 Predicted Label: {pred_label}")
        else:
            st.success(f"🎯 Predicted Label: {pred}")

else:
    st.info("👆 Please upload a CSV or Excel file to begin.")