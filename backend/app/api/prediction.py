import os
import shutil
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends,
)

from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import PredictionHistory
from app.services.model_service import model_service


router = APIRouter(
    prefix="/api",
    tags=["Medical Image Analysis"],
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads",
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


@router.post("/predict")
async def predict_xray(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename.",
        )

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported.",
        )

    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename,
    )

    try:

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        result = model_service.predict(
            file_path
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "Model returned invalid result."
            )

        prediction = result["prediction"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        normal_probability = (
            probabilities["NORMAL"]
        )

        pneumonia_probability = (
            probabilities["PNEUMONIA"]
        )

        # ====================================================
        # GRAD-CAM
        # ====================================================
        #
        # Disabled for cloud deployment because Grad-CAM
        # requires gradient computation and additional memory.
        #
        # The Grad-CAM implementation remains available in:
        #
        # app/services/gradcam_service.py
        #
        # ====================================================

        gradcam_url = None
        gradcam_status = "disabled_on_cloud"

        # ====================================================
        # DATABASE
        # ====================================================

        record = PredictionHistory(
            filename=str(file.filename),
            prediction=str(prediction),
            confidence=float(confidence),
            normal_probability=float(
                normal_probability
            ),
            pneumonia_probability=float(
                pneumonia_probability
            ),
            gradcam_path=None,
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        # ====================================================
        # RESPONSE
        # ====================================================

        return {
            "id": record.id,

            "filename": record.filename,

            "prediction": record.prediction,

            "confidence": float(
                record.confidence
            ),

            "probabilities": {
                "NORMAL": float(
                    record.normal_probability
                ),

                "PNEUMONIA": float(
                    record.pneumonia_probability
                ),
            },

            "gradcam_url": gradcam_url,

            "gradcam_status": gradcam_status,

            "created_at": (
                record.created_at.isoformat()
                if record.created_at
                else None
            ),

            "disclaimer": (
                "This AI output is for educational "
                "and research purposes only and is "
                "not a medical diagnosis."
            ),
        }

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{str(error)}"
            ),
        )

    finally:

        try:
            await file.close()
        except Exception:
            pass


# ============================================================
# PREDICTION HISTORY
# ============================================================

@router.get("/history")
def prediction_history(
    db: Session = Depends(get_db),
):

    return (
        db.query(PredictionHistory)
        .order_by(
            PredictionHistory.id.desc()
        )
        .all()
    )


# ============================================================
# PREDICTION DETAILS
# ============================================================

@router.get("/history/{prediction_id}")
def prediction_details(
    prediction_id: int,
    db: Session = Depends(get_db),
):

    record = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.id
            == prediction_id
        )
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    return record


# ============================================================
# DELETE PREDICTION
# ============================================================

@router.delete("/history/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
):

    record = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.id
            == prediction_id
        )
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    db.delete(record)
    db.commit()

    return {
        "message":
            "Prediction deleted successfully."
    }
