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
from app.services.gradcam_service import generate_gradcam


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
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# PREDICT X-RAY
# ============================================================

@router.post("/predict")
async def predict_xray(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

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
        # SAVE UPLOADED IMAGE
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
        # VERIFY FILE EXISTS
        # ====================================================

        if not os.path.exists(file_path):

            raise HTTPException(
                status_code=500,
                detail=(
                    "UPLOAD ERROR: "
                    "Saved image could not be found."
                )
            )


        if os.path.getsize(file_path) == 0:

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
                    f"{str(error)}"
                )
            )


        # ====================================================
        # VALIDATE MODEL RESPONSE
        # ====================================================

        if not isinstance(result, dict):

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "model_service.predict() did not "
                    "return a dictionary."
                )
            )


        if "prediction" not in result:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "prediction is missing from "
                    "model response."
                )
            )


        if "confidence" not in result:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "confidence is missing from "
                    "model response."
                )
            )


        probabilities = result.get(
            "probabilities"
        )


        if not isinstance(
            probabilities,
            dict
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "probabilities must be "
                    "a dictionary."
                )
            )


        if "NORMAL" not in probabilities:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "NORMAL probability is missing."
                )
            )


        if "PNEUMONIA" not in probabilities:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MODEL ERROR: "
                    "PNEUMONIA probability is missing."
                )
            )


        # ====================================================
        # GRAD-CAM
        # ====================================================

        try:

            gradcam_result = generate_gradcam(
                file_path
            )

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "GRADCAM ERROR: "
                    f"{str(error)}"
                )
            )


        # ====================================================
        # VALIDATE GRAD-CAM RESPONSE
        # ====================================================

        if not isinstance(
            gradcam_result,
            dict
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "GRADCAM ERROR: "
                    "generate_gradcam() did not "
                    "return a dictionary."
                )
            )


        gradcam_path = gradcam_result.get(
            "gradcam_path"
        )


        if not gradcam_path:

            raise HTTPException(
                status_code=500,
                detail=(
                    "GRADCAM ERROR: "
                    "gradcam_path is missing."
                )
            )


        gradcam_filename = os.path.basename(
            gradcam_path
        )


        gradcam_url = (
            f"/gradcam/{gradcam_filename}"
        )


        # ====================================================
        # DATABASE
        # ====================================================

        try:

            record = PredictionHistory(

                filename=file.filename,

                prediction=result[
                    "prediction"
                ],

                confidence=float(
                    result["confidence"]
                ),

                normal_probability=float(
                    probabilities["NORMAL"]
                ),

                pneumonia_probability=float(
                    probabilities["PNEUMONIA"]
                ),

                gradcam_path=gradcam_url
            )


            db.add(record)

            db.commit()

            db.refresh(record)


        except Exception as error:

            db.rollback()

            raise HTTPException(
                status_code=500,
                detail=(
                    "DATABASE ERROR: "
                    f"{str(error)}"
                )
            )


        # ====================================================
        # SUCCESS RESPONSE
        # ====================================================

        return {

            "id":
                record.id,

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
                "educational and research purposes "
                "only and is not a medical diagnosis."
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
                f"{str(error)}"
            )
        )


    finally:

        await file.close()


# ============================================================
# PREDICTION HISTORY
# ============================================================

@router.get("/history")
def prediction_history(
    db: Session = Depends(get_db)
):

    try:

        records = (
            db.query(PredictionHistory)
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
                f"{str(error)}"
            )
        )


# ============================================================
# SINGLE PREDICTION
# ============================================================

@router.get("/history/{prediction_id}")
def prediction_details(
    prediction_id: int,
    db: Session = Depends(get_db)
):

    try:

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


    except HTTPException:

        raise


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "HISTORY ERROR: "
                f"{str(error)}"
            )
        )


# ============================================================
# DELETE PREDICTION
# ============================================================

@router.delete("/history/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):

    try:

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


    except HTTPException:

        raise


    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "DELETE ERROR: "
                f"{str(error)}"
            )
        )
