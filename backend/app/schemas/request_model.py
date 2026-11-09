from pydantic import BaseModel, Field


class TranslatorRequest(BaseModel):
    medical_term: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Medical term to translate into patient-friendly language",
        example="Hypertension"
    )

    language: str = Field(
        default="English",
        description="Output language",
        example="English"
    )

    user_id: str = Field(
        default="anonymous",
        description="User identifier",
        example="USR001"
    )

    session_id: str = Field(
        default="session_001",
        description="Session identifier",
        example="SESSION001"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "medical_term": "Hypertension",
                "language": "English",
                "user_id": "USR001",
                "session_id": "SESSION001"
            }
        }
    }