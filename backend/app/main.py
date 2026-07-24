from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os

from app.api.prediction import router as prediction_router
from app.api.report import router as report_router


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUTS_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

GRADCAM_DIR = os.path.join(
    OUTPUTS_DIR,
    "gradcam"
)

UPLOADS_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)


# Create required directories
os.makedirs(
    OUTPUTS_DIR,
    exist_ok=True
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)

os.makedirs(
    UPLOADS_DIR,
    exist_ok=True
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description=(
        "AI-powered chest X-ray analysis platform using "
        "EfficientNet-B0, Grad-CAM explainability and "
        "Gemini-generated medical reports."
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
    allow_headers=["*"],
)


# ============================================================
# STATIC FILES
# ============================================================
# Makes Grad-CAM images accessible through:
#
# /outputs/gradcam/<filename>.jpg
#
# Example:
# https://your-backend.onrender.com/
# outputs/gradcam/gradcam_xxx.jpg
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(
        directory=OUTPUTS_DIR
    ),
    name="outputs"
)


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(
    prediction_router
)

app.include_router(
    report_router
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "name": (
            "Advanced AI Medical "
            "Intelligence Platform"
        ),
        "version": "1.0.0",
        "status": "running",
        "model": "EfficientNet-B0",
        "task": (
            "Chest X-ray "
            "Pneumonia Detection"
        ),
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
        "endpoints": {
            "prediction": "/api/predict",
            "history": "/api/history",
            "report": (
                "/api/generate-report/"
                "{prediction_id}"
            ),
            "documentation": "/docs",
            "health": "/health"
        },
        "disclaimer": (
            "This application is intended "
            "for educational and research "
            "purposes only. It is not a "
            "substitute for professional "
            "medical diagnosis."
        )
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": (
            "advanced-medical-ai-backend"
        ),
        "model": "EfficientNet-B0"
    }
