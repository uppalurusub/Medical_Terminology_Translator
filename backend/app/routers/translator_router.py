from fastapi import APIRouter, HTTPException
from app.config.settings import settings
logger = settings.get_logger()

from app.schemas.request_model import TranslatorRequest
from app.services.translator_service import TranslatorService

router = APIRouter(
    prefix="/translator",
    tags=["Medical Terminology Translator"]
)


@router.post("/translate")
async def translate_medical_term(
    request: TranslatorRequest
):
    logger.info(
        f"Translation request received "
        f"for term: {request.medical_term}"
    )

    try:

        response = TranslatorService.translate(
            medical_term=request.medical_term,
            language=request.language
        )

        logger.success(
            f"Translation completed "
            f"for term: {request.medical_term}"
        )

        return response

    except Exception as ex:

        logger.exception(
            f"Translation failed: {str(ex)}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )