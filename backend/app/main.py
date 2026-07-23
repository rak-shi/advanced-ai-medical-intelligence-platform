import os

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from app.database.database import Base, engine

# Import database models so SQLAlchemy knows about the tables
from app.database import models

# Import API routers
from app.api.prediction import router as prediction_router
from app.api.report import router as report_router


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

# Create database tables if they do not already exist
Base.metadata.create_all(
    bind=engine
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Advanced AI Medical Intelligence Platform",

    description=(
        "AI-powered chest X-ray analysis platform using "
        "EfficientNet-B0, Grad-CAM explainable AI, "
        "Gemini LLM-generated medical reports, "
        "and SQLite prediction history."
    ),

    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

# Allows the Streamlit frontend to communicate
# with the FastAPI backend.

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# GRAD-CAM OUTPUT DIRECTORY
# ============================================================

GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "gradcam"
)


os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# SERVE GRAD-CAM IMAGES
# ============================================================

app.mount(

    "/gradcam",

    StaticFiles(
        directory=GRADCAM_DIR
    ),

    name="gradcam"
)


# ============================================================
# REGISTER API ROUTERS
# ============================================================

# Deep-learning prediction APIs
app.include_router(
    prediction_router
)

# Gemini AI medical-report APIs
app.include_router(
    report_router
)


# ============================================================
# ROOT ENDPOINT
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
            "This application is intended for "
            "educational and research purposes only. "
            "It is not a substitute for professional "
            "medical diagnosis."
        )
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "service":
            "Advanced AI Medical Intelligence Platform",

        "model":
            "EfficientNet-B0"
    }