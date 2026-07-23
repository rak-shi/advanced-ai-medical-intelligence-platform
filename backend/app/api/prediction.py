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
from app.database.models import PredictionHistory

from app.services.model_service import model_service


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["Medical Image Analysis"]
)


# ============================================================
# DIRECTORIES
# ============================================================

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


# ============================================================
# ALLOWED EXTENSIONS
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@router.post("/predict")
async def predict_xray(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    file_path = None


    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename."
        )


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


    # --------------------------------------------------------
    # Generate unique filename
    # --------------------------------------------------------

    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )


    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename
    )


    try:

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        try:

            with open(
                file_path,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )


        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "UPLOAD ERROR: "
                    f"{str(error)}"
                )
            )


        # ====================================================
        # VALIDATE SAVED IMAGE
        # ====================================================

        if not os.path.exists(
            file_path
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "UPLOAD ERROR: "
                    "Uploaded image was not saved."
                )
            )


        if os.path.getsize(
            file_path
        ) == 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "UPLOAD ERROR: "
                    "Uploaded image is empty."
                )
            )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        try:

            result = model_service.predict(
                file_path
            )


        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    f"{type(error).__name__}: "
                    f"{str(error)}"
                )
            )


        # ====================================================
        # VALIDATE RESULT
        # ====================================================

        if not isinstance(
            result,
            dict
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "Prediction result is not "
                    "a dictionary."
                )
            )


        prediction = result.get(
            "prediction"
        )


        confidence = result.get(
            "confidence"
        )


        probabilities = result.get(
            "probabilities"
        )


        if prediction is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "prediction missing."
                )
            )


        if confidence is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "confidence missing."
                )
            )


        if not isinstance(
            probabilities,
            dict
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "probabilities missing "
                    "or invalid."
                )
            )


        # ====================================================
        # NORMAL / PNEUMONIA VALUES
        # ====================================================

        normal_probability = (
            probabilities.get(
                "NORMAL"
            )
        )


        pneumonia_probability = (
            probabilities.get(
                "PNEUMONIA"
            )
        )


        if normal_probability is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "NORMAL probability missing. "
                    f"Available classes: "
                    f"{list(probabilities.keys())}"
                )
            )


        if pneumonia_probability is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "PNEUMONIA probability missing. "
                    f"Available classes: "
                    f"{list(probabilities.keys())}"
                )
            )


        # ====================================================
        # GRAD-CAM
        #
        # TEMPORARILY DISABLED
        # ====================================================

        gradcam_url = None


        # ====================================================
        # DATABASE
        # ====================================================

        try:

            record = PredictionHistory(

                filename=str(
                    file.filename
                ),

                prediction=str(
                    prediction
                ),

                confidence=float(
                    confidence
                ),

                normal_probability=float(
                    normal_probability
                ),

                pneumonia_probability=float(
                    pneumonia_probability
                ),

                gradcam_path=gradcam_url
            )


            db.add(
                record
            )

            db.commit()

            db.refresh(
                record
            )


        except Exception as error:

            db.rollback()


            raise HTTPException(
                status_code=500,
                detail=(
                    "DATABASE ERROR: "
                    f"{type(error).__name__}: "
                    f"{str(error)}"
                )
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        return {

            "id":
                record.id,

            "filename":
                record.filename,

            "prediction":
                record.prediction,

            "confidence":
                float(
                    record.confidence
                ),

            "probabilities": {

                "NORMAL":
                    float(
                        record.normal_probability
                    ),

                "PNEUMONIA":
                    float(
                        record.pneumonia_probability
                    )
            },

            "gradcam_url":
                None,

            "gradcam_status":
                "temporarily_disabled",

            "created_at":
                (
                    record.created_at.isoformat()
                    if record.created_at
                    else None
                ),

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
            detail=(
                "UNEXPECTED ERROR: "
                f"{type(error).__name__}: "
                f"{str(error)}"
            )
        )


    finally:

        try:

            await file.close()

        except Exception:

            pass


# ============================================================
# HISTORY
# ============================================================

@router.get("/history")
def prediction_history(
    db: Session = Depends(get_db)
):

    try:

        records = (
            db.query(
                PredictionHistory
            )
            .order_by(
                PredictionHistory.id.desc()
            )
            .all()
        )


        return records


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "HISTORY ERROR: "
                f"{type(error).__name__}: "
                f"{str(error)}"
            )
        )


# ============================================================
# SINGLE RECORD
# ============================================================

@router.get(
    "/history/{prediction_id}"
)
def prediction_details(
    prediction_id: int,
    db: Session = Depends(get_db)
):

    try:

        record = (
            db.query(
                PredictionHistory
            )
            .filter(
                PredictionHistory.id
                == prediction_id
            )
            .first()
        )


        if not record:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Prediction not found."
                )
            )


        return record


    except HTTPException:

        raise


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "HISTORY ERROR: "
                f"{type(error).__name__}: "
                f"{str(error)}"
            )
        )


# ============================================================
# DELETE HISTORY
# ============================================================

@router.delete(
    "/history/{prediction_id}"
)
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):

    try:

        record = (
            db.query(
                PredictionHistory
            )
            .filter(
                PredictionHistory.id
                == prediction_id
            )
            .first()
        )


        if not record:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Prediction not found."
                )
            )


        db.delete(
            record
        )

        db.commit()


        return {
            "message":
                "Prediction deleted successfully."
        }


    except HTTPException:

        raise


    except Exception as error:

        db.rollback()


        raise HTTPException(
            status_code=500,
            detail=(
                "DELETE ERROR: "
                f"{type(error).__name__}: "
                f"{str(error)}"
            )
        )
