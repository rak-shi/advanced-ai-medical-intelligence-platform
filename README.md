# 🩻 Advanced AI Medical Intelligence Platform

An end-to-end AI-powered medical imaging application for **chest X-ray pneumonia detection**. The system combines deep learning, explainable AI, a FastAPI backend, a Streamlit user interface, SQLite-based prediction history, and Gemini-powered AI report generation.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

The Advanced AI Medical Intelligence Platform analyzes chest X-ray images and predicts whether an image belongs to:

- **NORMAL**
- **PNEUMONIA**

The system uses a trained **EfficientNet-B0** deep-learning model for image classification.

The application also integrates **Grad-CAM** for explainable AI and **Gemini** for generating an AI-assisted medical analysis report.

The complete application consists of:

- Deep-learning model
- Chest X-ray preprocessing
- Pneumonia classification
- Confidence and class probabilities
- Grad-CAM explainability
- AI-generated medical report
- Prediction history
- FastAPI REST API
- Streamlit web interface
- SQLite database
- Cloud deployment

---

## 🚀 Live Application

### Streamlit Frontend

https://advanced-ai-medical-intelligence-platform.streamlit.app

### FastAPI Backend

https://advanced-medical-ai-backend.onrender.com

> The backend is deployed using a free Render instance. The service may spin down after inactivity, so the first request can take additional time while the server starts.

---

## 🧠 AI Model

The medical imaging classifier is based on:

**EfficientNet-B0**

The trained model is stored at:

```text
backend/trained_model/pneumonia_efficientnet_b0.pth
```

The model performs binary chest X-ray classification:

```text
Chest X-ray
     │
     ▼
Image Preprocessing
     │
     ▼
EfficientNet-B0
     │
     ▼
Class Probabilities
     │
     ├── NORMAL
     │
     └── PNEUMONIA
```

---

## 📊 Model Performance

The trained model achieved approximately:

**Test Accuracy: 86.06%**

Training and evaluation outputs are available under:

```text
training/outputs/
```

including:

```text
accuracy_curve.png
loss_curve.png
confusion_matrix.png
evaluation_results.txt
```

---

## ✨ Main Features

### 🩻 Chest X-ray Analysis

Users can upload a chest X-ray image directly through the Streamlit interface.

The application displays image information such as:

- Filename
- Image format
- Image dimensions
- Image mode

The image is then sent to the FastAPI backend for analysis.

### 🧠 Pneumonia Prediction

The trained EfficientNet-B0 model generates:

- Predicted class
- Prediction confidence
- NORMAL probability
- PNEUMONIA probability
- Prediction ID

Example:

```text
Prediction: PNEUMONIA
Confidence: 98.83%

PNEUMONIA Probability: 98.83%
NORMAL Probability: 1.17%
```

### 🔥 Explainable AI with Grad-CAM

The project incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explainability.

Grad-CAM highlights regions of the chest X-ray that contributed to the neural network's prediction.

This makes the model's behavior more interpretable.

> Grad-CAM visualizations indicate regions influencing the model output and should not be interpreted as proof of pathology.

### 🤖 AI-Assisted Medical Report

The system integrates **Gemini** to generate a structured explanation from the AI prediction.

The generated report includes sections such as:

1. AI Analysis
2. Findings
3. Interpretation
4. Recommended Next Steps
5. Limitations and Disclaimer

Example:

```text
AI Analysis

Predicted Class: PNEUMONIA
Model Confidence: 98.83%

Findings

The deep-learning model evaluated the provided chest X-ray
and classified it under the category of PNEUMONIA.

Interpretation

The model output indicates that visual features in the image
closely align with patterns associated with pneumonia during
model training.
```

The generated text is explicitly presented as AI-assisted educational output rather than a clinical diagnosis.

### 📋 Prediction History

Prediction results can be stored in an **SQLite database**, allowing the application to maintain previous analysis records.

The Streamlit interface includes a Prediction History section for viewing stored predictions.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │       User           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Frontend   │
                    │    frontend/app.py   │
                    └──────────┬───────────┘
                               │
                          HTTP / REST
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FastAPI Backend    │
                    │ backend/app/main.py  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────────┐ ┌──────────────┐ ┌───────────────┐
     │ EfficientNet-B0│ │   Grad-CAM   │ │    Gemini     │
     │ Classification │ │ Explainability│ │ AI Reporting  │
     └────────┬───────┘ └──────────────┘ └───────────────┘
              │
              ▼
       ┌─────────────┐
       │   SQLite    │
       │  Database   │
       └─────────────┘
```

---

## 🔄 Application Workflow

```text
Upload Chest X-ray
        │
        ▼
Image Validation & Preprocessing
        │
        ▼
EfficientNet-B0 Inference
        │
        ▼
NORMAL / PNEUMONIA Prediction
        │
        ├──────────────► Confidence & Probabilities
        │
        ├──────────────► Grad-CAM Explainability
        │
        ├──────────────► SQLite Prediction Storage
        │
        └──────────────► Gemini AI Report
                                │
                                ▼
                     Streamlit Results Dashboard
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Model | EfficientNet-B0 |
| Computer Vision | OpenCV / Pillow |
| Explainable AI | Grad-CAM |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Gemini |
| Database | SQLite |
| API Communication | REST / Requests |
| Backend Deployment | Render |
| Frontend Deployment | Streamlit Community Cloud |
| Version Control | Git / GitHub |

---

## 📁 Project Structure

```text
advanced-ai-medical-intelligence-platform/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── prediction.py
│   │   │   └── report.py
│   │   │
│   │   ├── database/
│   │   │   ├── database.py
│   │   │   └── models.py
│   │   │
│   │   ├── schemas/
│   │   │   └── prediction.py
│   │   │
│   │   ├── services/
│   │   │   ├── gemini_service.py
│   │   │   ├── gradcam_service.py
│   │   │   └── model_service.py
│   │   │
│   │   └── main.py
│   │
│   ├── trained_model/
│   │   └── pneumonia_efficientnet_b0.pth
│   │
│   ├── .env.example
│   ├── test_gemini.py
│   └── test_prediction.py
│
├── frontend/
│   └── app.py
│
├── training/
│   ├── outputs/
│   │   ├── accuracy_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── evaluation_results.txt
│   │   └── loss_curve.png
│   │
│   └── train.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rak-shi/advanced-ai-medical-intelligence-platform.git
cd advanced-ai-medical-intelligence-platform
```

### 2. Create Virtual Environment

Windows:

```bash
py -3.11 -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create:

```text
backend/.env
```

Use `backend/.env.example` as the template.

For Gemini functionality, configure the required Gemini API key/environment variable according to the variable name used by the backend.

Do **not** commit the real `.env` file or API keys to GitHub.

---

## ▶️ Running the Backend

From the project root:

```bash
uvicorn app.main:app --app-dir backend --reload
```

Backend:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🖥️ Running the Frontend

Open another terminal and run:

```bash
streamlit run frontend/app.py
```

The application will normally open at:

```text
http://localhost:8501
```

For local development, the frontend can communicate with:

```text
http://127.0.0.1:8000
```

For the deployed application, it communicates with the Render backend.

---

## 🔌 API

The FastAPI backend exposes endpoints for medical-image prediction and AI report generation.

Example report endpoint:

```text
POST /api/generate-report/{prediction_id}
```

Example response:

```json
{
  "prediction_id": 2,
  "prediction": "PNEUMONIA",
  "confidence": 98.83,
  "medical_report": "AI-generated analysis...",
  "disclaimer": "This AI-assisted report is for educational and research purposes only and does not constitute a medical diagnosis."
}
```

Interactive API documentation is available through FastAPI Swagger UI at `/docs`.

---

## 📸 Application Screens

### Chest X-ray Upload

The Streamlit dashboard allows users to upload and preview a chest X-ray before running the classifier.

### Prediction Dashboard

After successful inference, the system can display:

```text
Prediction
PNEUMONIA

Confidence
99.44%

Prediction ID
4
```

along with class probabilities such as:

```text
NORMAL       0.56%
PNEUMONIA   99.44%
```

### Explainable AI

The application provides a dedicated **Explainable AI — Grad-CAM** section to help visualize the image regions contributing to the neural network prediction.

### System Information

The application sidebar displays the primary system components:

```text
Model:          EfficientNet-B0
Task:           Pneumonia Detection
Explainability: Grad-CAM
LLM:            Gemini
Database:       SQLite
Backend:        FastAPI
```

---

## ☁️ Deployment

The application uses separate frontend and backend deployments.

### Frontend

The Streamlit application is deployed on **Streamlit Community Cloud**.

### Backend

The FastAPI API and trained EfficientNet model are deployed as a **Render Web Service**.

Production backend command:

```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT
```

The frontend communicates with:

```text
https://advanced-medical-ai-backend.onrender.com
```

Because the current backend uses a free Render instance, cold starts and memory limitations may affect response time or availability.

---

## 🧪 Training

Model training is implemented in:

```text
training/train.py
```

The training pipeline generates evaluation artifacts including:

- Accuracy curve
- Loss curve
- Confusion matrix
- Evaluation results

The trained model is exported and used by the FastAPI inference service.

---

## 🔐 Security

Sensitive configuration such as API keys is excluded from version control.

The following types of files/directories are excluded through `.gitignore`:

```text
backend/.env
venv/
dataset/
backend/uploads/
backend/outputs/
*.db
__pycache__/
```

An `.env.example` file can be included to document the required environment configuration without exposing credentials.

---

## ⚠️ Limitations

This system has several important limitations:

- The model performs binary NORMAL/PNEUMONIA classification only.
- Model predictions depend on the training data and learned patterns.
- High confidence does not guarantee clinical correctness.
- Grad-CAM is an explanatory visualization and does not identify or prove pathology.
- The system does not have access to complete patient history, laboratory results, symptoms, or physical examination findings.
- Gemini-generated reports are explanatory AI outputs rather than medical reports issued by a healthcare professional.
- The deployed backend may experience cold-start delays and resource limitations on free hosting infrastructure.

---

## ⚕️ Medical Disclaimer

**This project is an educational and research prototype.**

The predictions, probabilities, Grad-CAM visualizations, and AI-generated reports produced by this application **do not constitute medical diagnoses or medical advice**.

The application is not intended to replace radiologists, physicians, or other qualified healthcare professionals.

Any medical-image interpretation and healthcare decision should be performed by an appropriately qualified medical professional using the patient's complete clinical information.

---

## 📦 Submission Components

The project repository contains the primary components required for submission:

- [x] Complete Source Code
- [x] Trained EfficientNet-B0 Model
- [x] GitHub Repository
- [x] README Documentation
- [x] requirements.txt
- [x] Training and Evaluation Code
- [x] Model Evaluation Outputs
- [x] Streamlit Frontend
- [x] FastAPI Backend
- [x] Live Frontend Deployment
- [x] Live Backend Deployment
- [ ] PDF Project Report
- [ ] Dockerfile — Not implemented

---

## 👩‍💻 Author

**Rakshitha Valipireddy**

GitHub: `rak-shi`

---

## 📄 License / Usage

This repository was developed as an educational AI/ML project.

The medical AI components should be treated as experimental and must not be used for clinical diagnosis or patient-care decisions.
