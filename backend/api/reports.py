from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import sys

from typing import Optional

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
    from the stored e-Netra grade.
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
                "e-Netra primary grade is "
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
            "from the e-Netra grade only."
        ),
        "senanur_used": None,
    }


from backend.services.i18n_service import (
    DR_LABELS,
    DR_INTERPRETATIONS,
    REFERRAL_TRANSLATIONS,
    QUALITY_STATUS_TRANSLATIONS,
    normalize_language,
)


def _build_summary(
    screening_data,
    decision,
    language: str = "English",
):
    """
    Build a deterministic report summary localized in the target language
    (English, Hindi, or Marathi).
    """
    norm_lang = normalize_language(language)

    dr_grade = screening_data.get(
        "dr_grade"
    )

    dr_label_raw = screening_data.get(
        "dr_label"
    )

    dr_confidence = screening_data.get(
        "dr_confidence"
    )

    quality_status_raw = screening_data.get(
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
            if norm_lang == "English"
            else ("कोई डी.आर. वर्गीकरण उपलब्ध नहीं है।" if norm_lang == "Hindi" else "कोणतेही डी.आर. वर्गीकरण उपलब्ध नाही.")
        )
        localized_label = dr_label_raw
    else:
        try:
            dr_grade_int = int(dr_grade)
            interpretation = DR_INTERPRETATIONS.get(dr_grade_int, {}).get(norm_lang, "")
            localized_label = DR_LABELS.get(dr_grade_int, {}).get(norm_lang, dr_label_raw)
        except Exception:
            interpretation = "An unrecognized DR grade was stored."
            localized_label = dr_label_raw

    raw_dec_val = decision.get("decision", "NON-REFERABLE")
    dec_meta = REFERRAL_TRANSLATIONS.get(raw_dec_val, REFERRAL_TRANSLATIONS.get("NON-REFERABLE"))
    localized_decision = dec_meta["label"].get(norm_lang, raw_dec_val)
    localized_decision_reason = dec_meta["reason"].get(norm_lang, decision.get("reason", ""))

    q_stat_key = str(quality_status_raw or "").lower()
    localized_q_status = QUALITY_STATUS_TRANSLATIONS.get(q_stat_key, {}).get(norm_lang, quality_status_raw)

    return {
        "language": norm_lang,
        "interpretation": interpretation,
        "referral_decision": localized_decision,
        "referral_decision_code": raw_dec_val,
        "referral_reason": localized_decision_reason,
        "referable": decision.get(
            "referable"
        ),
        "primary_model": (
            "e-Netra clean EfficientNet-B0"
        ),
        "primary_grade": dr_grade,
        "primary_label": localized_label,
        "primary_confidence": dr_confidence,
        "image_quality_status": localized_q_status,
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
    lang: Optional[str] = Query(default=None, description="Target language (English, Hindi, Marathi)"),
    language: Optional[str] = Query(default=None, description="Alternative target language parameter"),
    db: Session = Depends(get_db),
):
    """
    Return a structured report for one screening in the requested language.

    Example:
        GET /api/v1/reports/DR-20261004-001?lang=Hindi
        GET /api/v1/reports/DR-20261004-001?language=Marathi
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
    query_lang = lang or language
    stored_lang = evidence_dict.get("preferred_language") or evidence_dict.get("language")
    target_language = normalize_language(query_lang or stored_lang or "English")

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
            decision,
            language=target_language,
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
            "e-Netra structured screening report"
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
                {
                    "English": (
                        "Grad-CAM++ represents model attention and should not be "
                        "interpreted as proof of an individual lesion."
                    ),
                    "Hindi": (
                        "Grad-CAM++ एआई मॉडल के ध्यान क्षेत्र को दर्शाता है और इसे "
                        "व्यक्तिगत घाव के निश्चित प्रमाण के रूप में नहीं माना जाना चाहिए।"
                    ),
                    "Marathi": (
                        "Grad-CAM++ मॉडेलच्या लक्ष केंद्रित केलेल्या भागास दर्शवते आणि त्यास "
                        "प्रत्यक्ष जखमेचा निश्चित पुरावा मानू नये."
                    ),
                }.get(
                    target_language,
                    "Grad-CAM++ represents model attention."
                )
            ),
        },

        "images":
            images,

        "evidence":
            evidence,

        "gemini":
            gemini_result,

        "language":
            target_language,
    }


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@router.get(
    "/{screening_id}/pdf"
)
def download_screening_pdf(
    screening_id: str,
    lang: Optional[str] = Query(default=None, description="Target language for PDF (English, Hindi, Marathi)"),
    language: Optional[str] = Query(default=None, description="Alternative target language parameter"),
    db: Session = Depends(get_db),
):
    """
    Download a clinical PDF report for one screening in the target language.

    Example:
        GET /api/v1/reports/DR-20261004-001/pdf?lang=Hindi
        GET /api/v1/reports/DR-20261004-001/pdf?language=Marathi
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
    query_lang = lang or language
    stored_lang = evidence_dict.get("preferred_language") or evidence_dict.get("language")
    target_language = normalize_language(query_lang or stored_lang or "English")

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
            language=target_language,
        )
    except Exception as exc:
        print(f"PDF generation error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF report: {str(exc)}",
        )

    filename = f"drishti_report_{screening_id}_{target_language.lower()}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )