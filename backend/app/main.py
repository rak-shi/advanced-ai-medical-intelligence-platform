import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# ============================================================
# DATABASE
# ============================================================

from app.database.database import Base, engine

# IMPORTANT:
# Import SQLAlchemy models before create_all()
# so prediction_history is registered with Base.metadata.
from app.database.models import PredictionHistory


# ============================================================
# API ROUTERS
# ============================================================

from app.api.prediction import router as prediction_router
from app.api.report import router as report_router


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description=(
        "AI-powered chest X-ray analysis using EfficientNet-B0, "
        "Grad-CAM explainability, Gemini LLM, SQLite and FastAPI."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# main.py is located at:
#
# backend/app/main.py
#
# Therefore dirname(dirname(__file__)) gives:
#
# backend/

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# GRAD-CAM DIRECTORY
# ============================================================

GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "gradcam"
)

# Create the directory automatically if it does not exist.
os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# SERVE GRAD-CAM IMAGES
# ============================================================

# Files saved inside:
#
# backend/outputs/gradcam/
#
# will be accessible from:
#
# /gradcam/<filename>

app.mount(
    "/gradcam",
    StaticFiles(directory=GRADCAM_DIR),
    name="gradcam"
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(prediction_router)
app.include_router(report_router)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "Advanced AI Medical Intelligence Platform",
        "version": "1.0.0",
        "status": "running",
        "model": "EfficientNet-B0",
        "task": "Chest X-ray Pneumonia Detection",
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
            "health": "/health",
            "prediction": "/api/predict",
            "history": "/api/history",
            "report": "/api/generate-report/{prediction_id}",
            "gradcam": "/gradcam/{filename}",
            "documentation": "/docs"
        },
        "disclaimer": (
            "This application is intended for educational "
            "and research purposes only. It is not a substitute "
            "for professional medical diagnosis."
        )
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "backend": "FastAPI",
        "model": "EfficientNet-B0",
        "database": "SQLite",
        "explainability": "Grad-CAM",
        "llm": "Gemini",
        "gradcam_directory": "available"
    }
