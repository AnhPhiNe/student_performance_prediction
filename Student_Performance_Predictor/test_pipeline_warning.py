import joblib
import pandas as pd

PIPELINE_PATH = "models/hcmue_student_full_pipeline_v1_0.joblib"
RAW_FEATURES_PATH = "models/raw_feature_names.joblib"

pipeline = joblib.load(PIPELINE_PATH)
raw_feature_names = joblib.load(RAW_FEATURES_PATH)

if not isinstance(raw_feature_names, list):
    raw_feature_names = list(raw_feature_names)

sample = {
    "Hours_Studied": 5,
    "Attendance": 85,
    "Previous_Scores": 75,
    "Tutoring_Sessions": 2,
    "Sleep_Hours": 7,
    "Physical_Activity": 3,
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "Medium",
    "Extracurricular_Activities": "Yes",
    "Motivation_Level": "Medium",
    "Internet_Access": "Yes",
    "Family_Income": "Medium",
    "Teacher_Quality": "Medium",
    "School_Type": "Public",
    "Peer_Influence": "Neutral",
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
    "Gender": "Male"
}

df = pd.DataFrame([sample], columns=raw_feature_names)

print("Type of input:", type(df))
print("Columns:", df.columns.tolist())

pred = pipeline.predict(df)
print("Prediction:", pred)