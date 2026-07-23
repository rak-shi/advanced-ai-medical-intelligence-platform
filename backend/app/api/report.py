from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import PredictionHistory
from app.services.gemini_service import generate_medical_report


router = APIRouter(
    prefix="/api",
    tags=["AI Medical Report"]
)


@router.post("/generate-report/{prediction_id}")
def create_medical_report(
    prediction_id: int,
    db: Session = Depends(get_db)
):

    record = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.id == prediction_id)
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )

    try:

        report = generate_medical_report(
            prediction=record.prediction,
            confidence=record.confidence,
            normal_probability=record.normal_probability,
            pneumonia_probability=record.pneumonia_probability
        )

        record.medical_report = report

        db.commit()
        db.refresh(record)

        return {
            "prediction_id": record.id,
            "prediction": record.prediction,
            "confidence": record.confidence,
            "medical_report": report,
            "disclaimer": (
                "This AI-assisted report is for educational "
                "and research purposes only and does not "
                "constitute a medical diagnosis."
            )
        }

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate AI report: {str(error)}"
        )