from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):

    id: int

    filename: str

    prediction: str

    confidence: float

    normal_probability: float

    pneumonia_probability: float

    gradcam_url: Optional[str] = None

    medical_report: Optional[str] = None

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )