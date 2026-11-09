from datetime import datetime
from pydantic import BaseModel, Field


class TranslatorResponse(BaseModel):

    medical_term: str = Field(
        description="Original medical terminology"
    )

    plain_language: str = Field(
        description="Patient-friendly translation"
    )

    explanation: str = Field(
        description="Simple explanation"
    )

    severity: str = Field(
        description="Low, Moderate, High"
    )

    language: str = Field(
        default="English",
        description="Output language"
    )

    status: str = Field(
        default="success",
        description="Request status"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "medical_term": "Hypertension",
                "plain_language": "High blood pressure",
                "explanation": "Blood pressure is consistently higher than normal.",
                "severity": "Moderate",
                "language": "English",
                "status": "success",
                "timestamp": "2026-06-24T15:30:00"
            }
        }
    }