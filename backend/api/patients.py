from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import sys

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import inspect
from sqlalchemy.orm import Session


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from backend.database.database import get_db
from backend.database.models import Screening


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/v1/patients",
    tags=["Patients"],
)


# ============================================================
# SERIALIZATION HELPER
# ============================================================

def _serialize_value(value):
    """
    Convert common SQLAlchemy/Python values into
    JSON-friendly values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    return value


def _screening_to_dict(screening):
    """
    Convert the current Screening SQLAlchemy object to a
    JSON-friendly dictionary using only columns that actually
    exist in the installed model.
    """

    mapper = inspect(
        screening
    ).mapper

    result = {}

    for column in mapper.column_attrs:
        column_name = column.key

        value = getattr(
            screening,
            column_name,
            None,
        )

        result[column_name] = (
            _serialize_value(value)
        )

    return result


# ============================================================
# GET PATIENT SCREENING HISTORY
# ============================================================

@router.get(
    "/{user_id}/history"
)
def get_patient_history(
    user_id: str,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of screenings to return.",
    ),
    db: Session = Depends(get_db),
):
    """
    Return screening history for one patient/user.

    Example:
        GET /patients/USER123/history

    Optional:
        GET /patients/USER123/history?limit=50
    """

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required.",
        )

    try:
        screenings = (
            db.query(Screening)
            .filter(
                Screening.user_id
                == str(user_id)
            )
            .limit(limit)
            .all()
        )

    except Exception as exc:
        print(
            f"History query failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve screening history.",
        )

    history = [
        _screening_to_dict(
            screening
        )
        for screening in screenings
    ]

    return {
        "user_id": str(user_id),
        "total": len(history),
        "history": history,
    }


# ============================================================
# GET SINGLE SCREENING
# ============================================================

@router.get(
    "/{user_id}/history/{screening_id}"
)
def get_patient_screening(
    user_id: str,
    screening_id: str,
    db: Session = Depends(get_db),
):
    """
    Return one screening belonging to a specific patient.

    Example:
        GET /patients/USER123/history/DR-20261004-001
    """

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required.",
        )

    if not screening_id:
        raise HTTPException(
            status_code=400,
            detail="screening_id is required.",
        )

    try:
        screening = (
            db.query(Screening)
            .filter(
                Screening.user_id
                == str(user_id),
                Screening.screening_id
                == str(screening_id),
            )
            .first()
        )

    except Exception as exc:
        print(
            f"Screening history lookup failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve screening.",
        )

    if screening is None:
        raise HTTPException(
            status_code=404,
            detail="Screening not found.",
        )

    return {
        "user_id": str(user_id),
        "screening": _screening_to_dict(
            screening
        ),
    }