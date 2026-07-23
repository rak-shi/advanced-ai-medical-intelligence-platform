from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text
)

from app.database.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    filename = Column(
        String,
        nullable=False
    )

    prediction = Column(
        String,
        nullable=False
    )

    confidence = Column(
        Float,
        nullable=False
    )

    normal_probability = Column(
        Float
    )

    pneumonia_probability = Column(
        Float
    )

    gradcam_path = Column(
        String
    )

    medical_report = Column(
        Text,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )