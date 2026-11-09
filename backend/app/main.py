from fastapi import FastAPI
from app.config.settings import settings
logger = settings.get_logger()

from app.routers.translator_router import router as translator_router

app = FastAPI(
    title="Medical Terminology Translator API",
    description="Convert medical terminology into patient-friendly language",
    version="1.0.0"
)

# Register Router
app.include_router(translator_router)

logger.info("Routers Registered Successfully")




@app.get("/")
async def root():
    logger.info("Root Endpoint Accessed")

    return {
        "application": "Medical Terminology Translator",
        "status": "Running"
    }


@app.get("/health")
async def health_check():

    logger.info("Health Check Requested")

    return {
        "status": "Healthy"
    }