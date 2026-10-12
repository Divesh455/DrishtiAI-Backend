from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import sys

from typing import Optional

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
from backend.services.i18n_service import (
    DR_LABELS,
    DR_INTERPRETATIONS,
    REFERRAL_TRANSLATIONS,
    QUALITY_STATUS_TRANSLATIONS,
    normalize_language,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/v1/patients",
    tags=["Patients"],
)


def _enrich_screening_dict(screening_dict: dict, language: str = "English") -> dict:
    """Enrich a screening dictionary with localized labels and advice."""
    norm_lang = normalize_language(language)
    dr_grade = screening_dict.get("dr_grade")
    try:
        dr_grade_int = int(dr_grade) if dr_grade is not None else None
        loc_label = DR_LABELS.get(dr_grade_int, {}).get(norm_lang, screening_dict.get("dr_label"))
        loc_interp = DR_INTERPRETATIONS.get(dr_grade_int, {}).get(norm_lang, "")
    except Exception:
        loc_label = screening_dict.get("dr_label")
        loc_interp = ""

    ev = screening_dict.get("evidence")
    ev_dict = ev if isinstance(ev, dict) else {}
    dec_info = ev_dict.get("screening_decision") or {}
    raw_dec = dec_info.get("decision", "NON-REFERABLE")
    dec_meta = REFERRAL_TRANSLATIONS.get(raw_dec, REFERRAL_TRANSLATIONS["NON-REFERABLE"])
    loc_dec = dec_meta["label"].get(norm_lang, raw_dec)
    loc_reason = dec_meta["reason"].get(norm_lang, dec_info.get("reason", ""))

    q_stat = str(screening_dict.get("quality_status") or "").lower()
    loc_quality = QUALITY_STATUS_TRANSLATIONS.get(q_stat, {}).get(norm_lang, screening_dict.get("quality_status"))

    screening_dict["localized"] = {
        "language": norm_lang,
        "dr_label": loc_label,
        "interpretation": loc_interp,
        "referral_decision": loc_dec,
        "referral_reason": loc_reason,
        "quality_status": loc_quality,
    }
    return screening_dict


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
    lang: Optional[str] = Query(default=None, description="Target language (English, Hindi, Marathi)"),
    language: Optional[str] = Query(default=None, description="Alternative target language parameter"),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of screenings to return.",
    ),
    db: Session = Depends(get_db),
):
    """
    Return screening history for one patient/user with localized labels and summaries.

    Example:
        GET /api/v1/patients/USER123/history?lang=Hindi
        GET /api/v1/patients/USER123/history?language=Marathi
    """

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required.",
        )

    norm_lang = normalize_language(lang or language or "English")

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
        _enrich_screening_dict(_screening_to_dict(screening), language=norm_lang)
        for screening in screenings
    ]

    return {
        "user_id": str(user_id),
        "language": norm_lang,
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
    lang: Optional[str] = Query(default=None, description="Target language (English, Hindi, Marathi)"),
    language: Optional[str] = Query(default=None, description="Alternative target language parameter"),
    db: Session = Depends(get_db),
):
    """
    Return one screening belonging to a specific patient localized in the target language.

    Example:
        GET /api/v1/patients/USER123/history/DR-20261004-001?lang=Hindi
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

    norm_lang = normalize_language(lang or language or "English")

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

    enriched_screening = _enrich_screening_dict(
        _screening_to_dict(screening),
        language=norm_lang,
    )

    return {
        "user_id": str(user_id),
        "language": norm_lang,
        "screening": enriched_screening,
    }