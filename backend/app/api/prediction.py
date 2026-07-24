import os
import shutil
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends
)

from sqlalchemy.orm import Session

from app.database.database import get_db

from app.database.models import (
    PredictionHistory
)

from app.services.model_service import (
    model_service
)

from app.services.gradcam_service import (
    generate_gradcam
)


router = APIRouter(
    prefix="/api",
    tags=["Medical Image Analysis"]
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
    "uploads"
)


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


@router.post("/predict")
async def predict_xray(

    file: UploadFile = File(...),

    db: Session = Depends(get_db)
):

    extension = os.path.splitext(
        file.filename
    )[1].lower()


    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG "
                "images are supported."
            )
        )


    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )


    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename
    )


    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # ----------------------------------------
        # Prediction
        # ----------------------------------------

        result = model_service.predict(
            file_path
        )


        # ----------------------------------------
        # Grad-CAM
        # ----------------------------------------

        gradcam_result = generate_gradcam(
            file_path
        )


        gradcam_filename = os.path.basename(
            gradcam_result["gradcam_path"]
        )


        gradcam_url = (
            f"/gradcam/{gradcam_filename}"
        )


        # ----------------------------------------
        # Save history
        # ----------------------------------------

        record = PredictionHistory(

            filename=file.filename,

            prediction=result[
                "prediction"
            ],

            confidence=result[
                "confidence"
            ],

            normal_probability=(
                result["probabilities"][
                    "NORMAL"
                ]
            ),

            pneumonia_probability=(
                result["probabilities"][
                    "PNEUMONIA"
                ]
            ),

            gradcam_path=gradcam_url
        )


        db.add(record)

        db.commit()

        db.refresh(record)


        return {

            "id": record.id,

            "filename":
                record.filename,

            "prediction":
                record.prediction,

            "confidence":
                record.confidence,

            "probabilities": {
                "NORMAL":
                    record.normal_probability,

                "PNEUMONIA":
                    record.pneumonia_probability
            },

            "gradcam_url":
                gradcam_url,

            "created_at":
                record.created_at,

            "disclaimer": (
                "This AI output is for "
                "educational and research "
                "purposes only and is not "
                "a medical diagnosis."
            )
        }


    except HTTPException:
        raise


    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    finally:

        await file.close()


# ============================================================
# HISTORY
# ============================================================

@router.get("/history")
def prediction_history(

    db: Session = Depends(get_db)
):

    records = (
        db.query(PredictionHistory)
        .order_by(
            PredictionHistory.id.desc()
        )
        .all()
    )


    return records


# ============================================================
# SINGLE RECORD
# ============================================================

@router.get("/history/{prediction_id}")
def prediction_details(

    prediction_id: int,

    db: Session = Depends(get_db)
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
            detail="Prediction not found."
        )


    return record


# ============================================================
# DELETE HISTORY
# ============================================================

@router.delete("/history/{prediction_id}")
def delete_prediction(

    prediction_id: int,

    db: Session = Depends(get_db)
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
            detail="Prediction not found."
        )


    db.delete(record)

    db.commit()


    return {
        "message":
            "Prediction deleted successfully."
    }
