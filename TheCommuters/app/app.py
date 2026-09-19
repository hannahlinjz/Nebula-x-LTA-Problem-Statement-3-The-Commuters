import os
import tempfile

import joblib
import pandas as pd
import streamlit as st

from feature_extractor import extract_features_from_file


# Load model
APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, "rail_model.pkl")

model = joblib.load(MODEL_PATH)

label_map = {
    0: "Normal",
    1: "Side I",
    2: "Side II"
}


# Page settings
st.set_page_config(
    page_title="Rail Condition Monitoring",
    layout="wide"
)

st.title("Train Condition Monitoring")
st.subheader("Rail Corrugation Detection")

st.write(
    "Upload axle-box vibration CSV files to classify rail condition "
    "as Normal, Side I, or Side II corrugation."
)


# Upload files
uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)


# Run prediction
if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

    results = []
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as temp_file:

            temp_file.write(file.getbuffer())
            temp_path = temp_file.name

        try:
            # Feature extraction
            features = extract_features_from_file(
                temp_path
            ).reshape(1, -1)

            # Real model prediction
            pred_code = model.predict(features)[0]

            pred_label = label_map[pred_code]

            results.append({
                "file_id": file.name,
                "prediction": pred_label
            })

        finally:
            os.remove(temp_path)

        progress_bar.progress(
            (i + 1) / len(uploaded_files)
        )


    # Show results
    results_df = pd.DataFrame(results)

    st.subheader("Diagnostic Results")

    st.dataframe(
        results_df,
        use_container_width=True
    )


    # Summary
    st.subheader("Summary")

    st.write(
        results_df["prediction"].value_counts()
    )


    # Download CSV
    csv_data = (
        results_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="Download rail_predictions.csv",
        data=csv_data,
        file_name="rail_predictions.csv",
        mime="text/csv"
    )