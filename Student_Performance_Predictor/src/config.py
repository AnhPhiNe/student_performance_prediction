# src/config.py
import os

# =========================================================
# 1) ĐƯỜNG DẪN THƯ MỤC / FILE
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSS_DIR = os.path.join(ASSETS_DIR, "css")
ICONS_DIR = os.path.join(ASSETS_DIR, "icons")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(DATA_DIR, "Student_Performance.csv")

PIPELINE_PATH = os.path.join(MODELS_DIR, "hcmue_student_full_pipeline_v1_0.joblib")
CORE_MODEL_PATH = os.path.join(MODELS_DIR, "ridge_core_model.joblib")
RAW_FEATURES_PATH = os.path.join(MODELS_DIR, "raw_feature_names.joblib")
RAW_SURVIVORS_PATH = os.path.join(MODELS_DIR, "raw_survivors.joblib")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_hyperparameters.json")

CSS_PATH = os.path.join(CSS_DIR, "styles.css")


# =========================================================
# 2) TÊN CỘT HIỂN THỊ THÂN THIỆN CHO NGƯỜI DÙNG
# =========================================================
FRIENDLY_LABELS = {
    "Hours_Studied": "Hours Studied per Day",
    "Attendance": "Attendance (%)",
    "Previous_Scores": "Previous Exam Score",
    "Tutoring_Sessions": "Tutoring Sessions per Month",
    "Sleep_Hours": "Sleep Hours per Night",
    "Physical_Activity": "Physical Activity (hours/week)",
    "Parental_Involvement": "Parental Involvement",
    "Access_to_Resources": "Access to Learning Resources",
    "Extracurricular_Activities": "Extracurricular Activities",
    "Motivation_Level": "Motivation Level",
    "Internet_Access": "Internet Access",
    "Family_Income": "Family Income",
    "Teacher_Quality": "Teacher Quality",
    "School_Type": "School Type",
    "Peer_Influence": "Peer Influence",
    "Learning_Disabilities": "Learning Disabilities",
    "Parental_Education_Level": "Parental Education Level",
    "Distance_from_Home": "Distance from Home",
    "Gender": "Gender",
    "Exam_Score": "Exam Score",
}


# =========================================================
# 3) CÁC GIÁ TRỊ CHO CỘT PHÂN LOẠI
# =========================================================
CATEGORICAL_OPTIONS = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["No", "Yes"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Gender": ["Male", "Female"],
}


# =========================================================
# 4) GIÁ TRỊ MẶC ĐỊNH KHI TẠO FORM
# =========================================================
DEFAULT_VALUES = {
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
    "Gender": "Male",
}


# =========================================================
# 5) CÁC HỒ SƠ MẪU ĐỂ DEMO NHANH
# =========================================================
SAMPLE_PROFILES = {
    "Balanced Student": {
        "Hours_Studied": 5,
        "Attendance": 88,
        "Previous_Scores": 78,
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
        "Distance_from_Home": "Moderate",
        "Gender": "Male",
    },
    "High Performer": {
        "Hours_Studied": 8,
        "Attendance": 96,
        "Previous_Scores": 91,
        "Tutoring_Sessions": 4,
        "Sleep_Hours": 7,
        "Physical_Activity": 4,
        "Parental_Involvement": "High",
        "Access_to_Resources": "High",
        "Extracurricular_Activities": "Yes",
        "Motivation_Level": "High",
        "Internet_Access": "Yes",
        "Family_Income": "High",
        "Teacher_Quality": "High",
        "School_Type": "Private",
        "Peer_Influence": "Positive",
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "Postgraduate",
        "Distance_from_Home": "Near",
        "Gender": "Female",
    },
    "At-Risk Student": {
        "Hours_Studied": 2,
        "Attendance": 62,
        "Previous_Scores": 55,
        "Tutoring_Sessions": 0,
        "Sleep_Hours": 5,
        "Physical_Activity": 1,
        "Parental_Involvement": "Low",
        "Access_to_Resources": "Low",
        "Extracurricular_Activities": "No",
        "Motivation_Level": "Low",
        "Internet_Access": "No",
        "Family_Income": "Low",
        "Teacher_Quality": "Low",
        "School_Type": "Public",
        "Peer_Influence": "Negative",
        "Learning_Disabilities": "Yes",
        "Parental_Education_Level": "High School",
        "Distance_from_Home": "Far",
        "Gender": "Male",
    },
}


# =========================================================
# 6) KHOẢNG GIÁ TRỊ HỢP LỆ CHO BIẾN SỐ
# =========================================================
NUMERIC_RANGES = {
    "Hours_Studied": (0, 12),
    "Attendance": (0, 100),
    "Previous_Scores": (0, 100),
    "Tutoring_Sessions": (0, 20),
    "Sleep_Hours": (0, 15),
    "Physical_Activity": (0, 20),
}


# =========================================================
# 7) CÁC NHÓM INPUT ĐỂ CHIA FORM CHO ĐẸP
# =========================================================
FORM_GROUPS = {
    "Academic Habits": [
        "Hours_Studied",
        "Attendance",
        "Previous_Scores",
        "Tutoring_Sessions",
    ],
    "Lifestyle": [
        "Sleep_Hours",
        "Physical_Activity",
        "Motivation_Level",
        "Extracurricular_Activities",
    ],
    "Family & Learning Environment": [
        "Parental_Involvement",
        "Access_to_Resources",
        "Internet_Access",
        "Family_Income",
        "Parental_Education_Level",
    ],
    "School & Personal Context": [
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Learning_Disabilities",
        "Distance_from_Home",
        "Gender",
    ],
}