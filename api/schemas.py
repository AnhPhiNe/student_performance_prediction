from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentProfileRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
            }
        }
    )

    Hours_Studied: float = Field(ge=0, le=80)
    Attendance: float = Field(ge=0, le=100)
    Previous_Scores: float = Field(ge=0, le=100)
    Tutoring_Sessions: int = Field(ge=0, le=20)
    Sleep_Hours: float = Field(ge=0, le=15)
    Physical_Activity: float = Field(ge=0, le=20)
    Parental_Involvement: Literal["Low", "Medium", "High"]
    Access_to_Resources: Literal["Low", "Medium", "High"]
    Extracurricular_Activities: Literal["No", "Yes"]
    Motivation_Level: Literal["Low", "Medium", "High"]
    Internet_Access: Literal["No", "Yes"]
    Family_Income: Literal["Low", "Medium", "High"]
    Teacher_Quality: Literal["Low", "Medium", "High"]
    School_Type: Literal["Public", "Private"]
    Peer_Influence: Literal["Negative", "Neutral", "Positive"]
    Learning_Disabilities: Literal["No", "Yes"]
    Parental_Education_Level: Literal["High School", "College", "Postgraduate"]
    Distance_from_Home: Literal["Near", "Moderate", "Far"]
    Gender: Literal["Male", "Female"]


class PredictionResponse(BaseModel):
    predicted_score: float
    predicted_band: str
    recommendations: list[str]
    warnings: list[str] = Field(default_factory=list)


class BatchPredictionRequest(BaseModel):
    records: list[StudentProfileRequest] = Field(min_length=1)


class BatchRecordPrediction(BaseModel):
    row_id: int
    predicted_score: float
    predicted_band: str


class BatchPredictionResponse(BaseModel):
    count: int
    average_score: float
    predictions: list[BatchRecordPrediction]
    warnings: list[str] = Field(default_factory=list)


class MetadataResponse(BaseModel):
    model_name: str
    target: str
    features: list[str]
    selected_features: list[str]
    metrics: dict[str, float]
    training_date: str | None = None
