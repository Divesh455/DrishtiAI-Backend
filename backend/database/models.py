from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.database import Base


class Screening(Base):
    """
    Stores one complete DrishtiAI screening result.

    One user can have many Screening records.
    """

    __tablename__ = "screenings"

    # -----------------------------------------------------
    # Primary identity
    # -----------------------------------------------------

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    screening_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    # -----------------------------------------------------
    # DR Classification
    # -----------------------------------------------------

    dr_grade: Mapped[Optional[int]] = mapped_column(
        SmallInteger,
        nullable=True,
    )

    dr_label: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    dr_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    class_probabilities: Mapped[Optional[Any]] = mapped_column(
        JSONB,
        nullable=True,
    )

    # -----------------------------------------------------
    # Image Quality / Fundus Suitability
    # -----------------------------------------------------

    quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    quality_status: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
    )

    fundus_suitability: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    fundus_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # -----------------------------------------------------
    # Lesion Detection
    # -----------------------------------------------------

    lesion_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    lesion_details: Mapped[Optional[Any]] = mapped_column(
        JSONB,
        nullable=True,
    )

    # -----------------------------------------------------
    # Evidence / Explainability
    # -----------------------------------------------------

    evidence: Mapped[Optional[Any]] = mapped_column(
        JSONB,
        nullable=True,
    )

    # -----------------------------------------------------
    # Cloudinary Images
    # -----------------------------------------------------

    original_image_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    heatmap_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    gradcam_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    lesion_overlay_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # -----------------------------------------------------
    # Screening Status
    # -----------------------------------------------------

    screening_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="completed",
        server_default="completed",
    )

    def __repr__(self) -> str:
        return (
            f"<Screening("
            f"screening_id={self.screening_id!r}, "
            f"user_id={self.user_id!r}, "
            f"dr_grade={self.dr_grade!r}"
            f")>"
        )