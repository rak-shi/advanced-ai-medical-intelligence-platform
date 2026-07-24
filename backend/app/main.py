from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# DATABASE
# ============================================================

from app.database.database import Base, engine

# IMPORTANT:
# Import database models BEFORE create_all().
# This allows SQLAlchemy to register the prediction_history
# table before creating the SQLite database tables.
from app.database.models import PredictionHistory


# ============================================================
# API ROUTERS
# ============================================================

from app.api.prediction import router as prediction_router
from app.api.report import router as report_router


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

# Creates prediction_history and any other registered tables
# if they do not already exist.
Base.metadata.create_all(bind=engine)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description=(
        "AI-powered chest X-ray analysis platform using "
        "EfficientNet-B0, Grad-CAM explainability, Gemini LLM, "
        "SQLite, and FastAPI."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

# Allows the Streamlit frontend to communicate with the
# deployed FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REGISTER ROUTERS
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
        "disclaimer": (
            "This application is intended for educational "
            "and research purposes only. It is not a "
            "substitute for professional medical diagnosis."
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
        "llm": "Gemini"
    }
