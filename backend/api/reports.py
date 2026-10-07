from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import sys

from fastapi import APIRouter, Depends, HTTPException, Query, Response
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
from backend.services.pdf_service import generate_screening_pdf
from backend.services.patient_service import fetch_patient_profile
from backend.services.gemini_service import GeminiReportService


# ============================================================
# SERVICES & ROUTER
# ============================================================

gemini_service = GeminiReportService()

router = APIRouter(
    prefix="/api/v1/reports",
    tags=["Reports"],
)


# ============================================================
# SERIALIZATION
# ============================================================

def _serialize_value(value):
    """
    Convert common Python/SQLAlchemy values into
    JSON-compatible values.
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
    Read only the columns that actually exist in the
    current Screening model.
    """

    mapper = inspect(
        screening
    ).mapper

    result = {}

    for column in mapper.column_attrs:
        name = column.key

        value = getattr(
            screening,
            name,
            None,
        )

        result[name] = _serialize_value(
            value
        )

    return result


# ============================================================
# HELPERS
# ============================================================

def _safe_dict(value):
    """
    Return a dictionary when a stored value is JSON-like.
    """

    if isinstance(value, dict):
        return value

    return {}


def _extract_screening_decision(
    screening_data,
):
    """
    Read the locked referral decision from the existing
    evidence JSON.

    For older screening rows that do not contain the newer
    screening_decision field, derive a fallback decision
    from the stored DrishtiAI grade.
    """

    evidence = _safe_dict(
        screening_data.get(
            "evidence"
        )
    )

    stored_decision = evidence.get(
        "screening_decision"
    )

    if isinstance(
        stored_decision,
        dict,
    ):
        return stored_decision

    dr_grade = screening_data.get(
        "dr_grade"
    )

    if dr_grade is None:
        return {
            "decision": "UNKNOWN",
            "referable": None,
            "reason": (
                "No DR grade is stored."
            ),
            "senanur_used": None,
        }

    dr_grade = int(
        dr_grade
    )

    if dr_grade >= 2:
        return {
            "decision": "REFER",
            "referable": True,
            "reason": (
                "DrishtiAI primary grade is "
                f"{dr_grade}, which is referable "
                "(Grade >= 2)."
            ),
            "senanur_used": False,
        }

    return {
        "decision": "NON-REFERABLE",
        "referable": False,
        "reason": (
            "This screening record predates the "
            "stored Senanur safety decision, so "
            "the fallback decision is derived "
            "from the DrishtiAI grade only."
        ),
        "senanur_used": None,
    }


def _build_summary(
    screening_data,
    decision,
):
    """
    Build a deterministic report summary.

    Gemini is intentionally not called here. The endpoint
    returns structured facts from the stored screening record.
    """

    dr_grade = screening_data.get(
        "dr_grade"
    )

    dr_label = screening_data.get(
        "dr_label"
    )

    dr_confidence = screening_data.get(
        "dr_confidence"
    )

    quality_status = screening_data.get(
        "quality_status"
    )

    lesion_count = screening_data.get(
        "lesion_count"
    )

    screening_status = screening_data.get(
        "screening_status"
    )

    if dr_grade is None:
        interpretation = (
            "No DR classification is available."
        )

    elif int(dr_grade) == 0:
        interpretation = (
            "No diabetic retinopathy was "
            "classified by the primary DrishtiAI "
            "model."
        )

    elif int(dr_grade) == 1:
        interpretation = (
            "Mild diabetic retinopathy was "
            "classified by the primary DrishtiAI "
            "model."
        )

    elif int(dr_grade) == 2:
        interpretation = (
            "Moderate diabetic retinopathy was "
            "classified by the primary DrishtiAI "
            "model."
        )

    elif int(dr_grade) == 3:
        interpretation = (
            "Severe diabetic retinopathy was "
            "classified by the primary DrishtiAI "
            "model."
        )

    elif int(dr_grade) == 4:
        interpretation = (
            "Proliferative diabetic retinopathy "
            "was classified by the primary "
            "DrishtiAI model."
        )

    else:
        interpretation = (
            "An unrecognized DR grade was stored."
        )

    return {
        "interpretation": interpretation,
        "referral_decision": decision.get(
            "decision"
        ),
        "referable": decision.get(
            "referable"
        ),
        "primary_model": (
            "DrishtiAI clean EfficientNet-B0"
        ),
        "primary_grade": dr_grade,
        "primary_label": dr_label,
        "primary_confidence": dr_confidence,
        "image_quality_status": quality_status,
        "detected_lesion_count": lesion_count,
        "screening_status": screening_status,
    }


# ============================================================
# GET SCREENING REPORT
# ============================================================

@router.get(
    "/{screening_id}"
)
def get_screening_report(
    screening_id: str,
    lang: str = Query(default="English", description="Target language for report narrative generation (Primary: English, Hindi, Marathi)"),
    db: Session = Depends(get_db),
):
    """
    Return a structured report for one screening.

    Example:
        GET /api/v1/reports/DR-20261004-001?lang=Hindi
    """

    if not screening_id:
        raise HTTPException(
            status_code=400,
            detail="screening_id is required.",
        )

    try:
        screening = (
            db.query(Screening)
            .filter(
                Screening.screening_id
                == str(screening_id)
            )
            .first()
        )

    except Exception as exc:
        print(
            f"Report query failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve screening "
                "for report generation."
            ),
        )

    if screening is None:
        raise HTTPException(
            status_code=404,
            detail="Screening not found.",
        )

    screening_data = (
        _screening_to_dict(
            screening
        )
    )

    user_id = screening_data.get("user_id")
    patient_info = fetch_patient_profile(user_id)

    evidence_dict = _safe_dict(screening_data.get("evidence"))
    target_language = lang if lang != "English" else (evidence_dict.get("preferred_language") or evidence_dict.get("language") or lang)

    gemini_result = gemini_service.generate_report_narrative(
        screening_data=screening_data,
        patient_info=patient_info,
        language=target_language,
    )

    decision = (
        _extract_screening_decision(
            screening_data
        )
    )

    summary = (
        _build_summary(
            screening_data,
            decision
        )
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    quality = {
        "score":
            screening_data.get(
                "quality_score"
            ),
        "status":
            screening_data.get(
                "quality_status"
            ),
        "fundus_suitability":
            screening_data.get(
                "fundus_suitability"
            ),
        "fundus_confidence":
            screening_data.get(
                "fundus_confidence"
            ),
    }

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification = {
        "grade":
            screening_data.get(
                "dr_grade"
            ),
        "label":
            screening_data.get(
                "dr_label"
            ),
        "confidence":
            screening_data.get(
                "dr_confidence"
            ),
        "probabilities":
            screening_data.get(
                "class_probabilities"
            ),
    }

    # --------------------------------------------------------
    # Lesions
    # --------------------------------------------------------

    lesions = {
        "count":
            screening_data.get(
                "lesion_count"
            ),
        "details":
            screening_data.get(
                "lesion_details"
            ),
    }

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    images = {
        "original_url":
            screening_data.get(
                "original_image_url"
            ),
        "heatmap_url":
            screening_data.get(
                "heatmap_url"
            ),
        "gradcam_url":
            screening_data.get(
                "gradcam_url"
            ),
        "lesion_overlay_url":
            screening_data.get(
                "lesion_overlay_url"
            ),
    }

    # --------------------------------------------------------
    # Stored evidence
    # --------------------------------------------------------

    evidence = screening_data.get(
        "evidence"
    )

    # --------------------------------------------------------
    # Structured report
    # --------------------------------------------------------

    return {
        "report_version": "1.0",

        "report_type": (
            "DrishtiAI structured screening report"
        ),

        "screening": {
            "screening_id":
                screening_data.get(
                    "screening_id"
                ),
            "user_id":
                screening_data.get(
                    "user_id"
                ),
            "status":
                screening_data.get(
                    "screening_status"
                ),
        },

        "patient":
            patient_info,

        "summary":
            summary,

        "quality":
            quality,

        "classification":
            classification,

        "screening_decision":
            decision,

        "lesions":
            lesions,

        "explainability": {
            "method":
                "Grad-CAM++",
            "heatmap_url":
                screening_data.get(
                    "heatmap_url"
                ),
            "overlay_url":
                screening_data.get(
                    "gradcam_url"
                ),
            "note": (
                "Grad-CAM++ represents model "
                "attention and should not be "
                "interpreted as proof of an "
                "individual lesion."
            ),
        },

        "images":
            images,

        "evidence":
            evidence,

        "gemini":
            gemini_result,
    }


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@router.get(
    "/{screening_id}/pdf"
)
def download_screening_pdf(
    screening_id: str,
    lang: str = Query(default="English", description="Target language for report narrative generation (Primary: English, Hindi, Marathi)"),
    db: Session = Depends(get_db),
):
    """
    Download a clinical PDF report for one screening.

    Example:
        GET /api/v1/reports/DR-20261004-001/pdf?lang=Hindi
    """

    if not screening_id:
        raise HTTPException(
            status_code=400,
            detail="screening_id is required.",
        )

    try:
        screening = (
            db.query(Screening)
            .filter(
                Screening.screening_id
                == str(screening_id)
            )
            .first()
        )
    except Exception as exc:
        print(f"PDF query failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve screening for PDF generation.",
        )

    if screening is None:
        raise HTTPException(
            status_code=404,
            detail="Screening not found.",
        )

    screening_data = _screening_to_dict(screening)

    user_id = screening_data.get("user_id")
    patient_info = fetch_patient_profile(user_id)

    evidence_dict = _safe_dict(screening_data.get("evidence"))
    target_language = lang if lang != "English" else (evidence_dict.get("preferred_language") or evidence_dict.get("language") or lang)

    gemini_result = gemini_service.generate_report_narrative(
        screening_data=screening_data,
        patient_info=patient_info,
        language=target_language,
    )

    try:
        pdf_bytes = generate_screening_pdf(
            screening_data,
            patient_info=patient_info,
            gemini_narrative=gemini_result,
        )
    except Exception as exc:
        print(f"PDF generation error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF report: {str(exc)}",
        )

    filename = f"drishti_report_{screening_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )