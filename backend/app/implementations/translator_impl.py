from app.config.settings import settings
logger = settings.get_logger()


from pydantic import BaseModel, Field

from langchain_core.output_parsers import (
    PydanticOutputParser
)

logger.info(
    "Initializing Medical Translation Schema"
)


class MedicalTranslation(BaseModel):

    medical_term: str = Field(
        description="Original medical terminology"
    )

    plain_language: str = Field(
        description="Patient-friendly translation"
    )

    explanation: str = Field(
        description="Simple explanation for patients"
    )

    severity: str = Field(
        description="Low, Moderate, High"
    )


logger.info(
    "Creating Pydantic Output Parser"
)

translation_parser = PydanticOutputParser(
    pydantic_object=MedicalTranslation
)

logger.success(
    "Medical Translation Parser Initialized Successfully"
)