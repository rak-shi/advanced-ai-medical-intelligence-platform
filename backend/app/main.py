import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.prediction import router as prediction_router
from app.api.routes.report import router as report_router

from app.database.database import Base, engine


# ============================================================
# DATABASE
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description=(
        "AI-powered chest X-ray pneumonia classification "
        "with EfficientNet-B0, Grad-CAM explainability "
        "and Gemini AI-assisted medical reports."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

GRADCAM_DIR = os.path.join(
    OUTPUT_DIR,
    "gradcam"
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(
        directory=OUTPUT_DIR
    ),
    name="outputs"
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(
    prediction_router
)

app.include_router(
    report_router
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name":
            "Advanced AI Medical Intelligence Platform",

        "version":
            "1.0.0",

        "status":
            "running",

        "model":
            "EfficientNet-B0",

        "task":
            "Chest X-ray Pneumonia Detection",

        "classes": [
            "NORMAL",
            "PNEUMONIA"
        ],

        "features": [
            "Deep Learning Prediction",
            "Grad-CAM Explainable AI",
            "Gemini AI Medical Report",
            "SQLite Prediction History",
            "REST API"
        ],

        "disclaimer": (
            "This application is intended "
            "for educational and research "
            "purposes only. It is not a "
            "substitute for professional "
            "medical diagnosis."
        )
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "EfficientNet-B0",
        "model_loaded":
            model_service_is_available()
    }


def model_service_is_available():

    try:

        from app.services.model_service import (
            model_service
        )

        return (
            model_service.model
            is not None
        )

    except Exception:

        return False
