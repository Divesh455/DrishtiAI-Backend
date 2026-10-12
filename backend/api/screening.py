from io import BytesIO

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from PIL import Image

from sqlalchemy.orm import Session

from backend.database.database import get_db

from backend.services.quality_service import (
    QualityService,
)

from backend.services.screening_service import (
    ScreeningService,
)
from backend.services.patient_service import prefetch_patient_profile
from backend.services.gemini_service import GeminiReportService
from backend.services.i18n_service import (
    normalize_language,
    localize_screening_response,
)


router = APIRouter(
    prefix="/api/v1/screening",
    tags=["Screening"],
)


# ============================================================
# SERVICES
# ============================================================

quality_service = QualityService()

screening_service = ScreeningService()

gemini_service = GeminiReportService()


# ============================================================
# STATUS
# ============================================================

@router.get("/status")
def screening_status():

    return {
        "service": "e-Netra Screening API",
        "status": "ready",
        "database": "Neon PostgreSQL",
        "storage": "Cloudinary",
    }


# ============================================================
# QUALITY CHECK
# ============================================================

@router.post("/quality")
def check_quality(
    file: UploadFile = File(...),
):

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image type. "
                "Allowed: JPEG, PNG, WEBP."
            ),
        )

    try:

        contents = file.file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read image.",
        )

    try:

        result = (
            quality_service.check_pil_image(
                image
            )
        )

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            **result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Quality check failed: {str(exc)}"
            ),
        )


# ============================================================
# COMPLETE SCREENING
# ============================================================

@router.post("/screen")
def screen_image(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    language: str = Form(default="English", description="Target language for screening report generation (Primary: English, Hindi, Marathi)"),
    db: Session = Depends(get_db),
):

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    # --------------------------------------------------------
    # Validate image type
    # --------------------------------------------------------

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image type. "
                "Allowed: JPEG, PNG, WEBP."
            ),
        )

    # --------------------------------------------------------
    # Validate user ID & prefetch patient profile asynchronously
    # --------------------------------------------------------

    if not user_id.strip():

        raise HTTPException(
            status_code=400,
            detail="user_id is required.",
        )

    # Parallel prefetch to warm up cache
    prefetch_patient_profile(user_id)

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        contents = file.file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read image.",
        )

    # --------------------------------------------------------
    # Technical + fundus quality check
    # --------------------------------------------------------

    try:

        quality_result = (
            quality_service.check_pil_image(
                image
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Quality check failed: {str(exc)}"
            ),
        )

    # --------------------------------------------------------
    # Reject unsuitable image & record in DB
    # --------------------------------------------------------

    norm_lang = normalize_language(language)

    if not quality_result.get(
        "screening_ready",
        False,
    ):

        screening_id = (
            screening_service.output_service.create_screening_id()
        )

        try:
            screening_service._save_screening(
                db=db,
                user_id=user_id,
                screening_id=screening_id,
                quality_result=quality_result,
                classification_result=None,
                lesion_result=None,
                evidence=None,
                images=None,
                screening_status="rejected",
            )
        except Exception as save_exc:
            print(f"Failed to record rejected screening in DB: {save_exc}")

        reject_msg = {
            "English": "Image rejected due to poor technical quality or unsuitable fundus view. Please capture a clear fundus image.",
            "Hindi": "खराब तकनीकी गुणवत्ता या अनुपयुक्त दृश्य के कारण छवि अस्वीकृत कर दी गई। कृपया स्पष्ट फंडस छवि पुनः अपलोड करें।",
            "Marathi": "कमी तांत्रिक गुणवत्ता किंवा अयोग्य दृश्यामुळे प्रतिमा नाकारली गेली. कृपया स्पष्ट फंडस प्रतिमा पुन्हा अपलोड करा.",
        }.get(norm_lang, "Image rejected due to poor quality.")

        return {
            "filename": file.filename,
            "user_id": user_id,
            "screening_id": screening_id,
            "screening_status": "rejected",
            "language": norm_lang,
            "message": reject_msg,
            "quality": quality_result,
            "screening": None,
            "ai_report": None,
        }

    # --------------------------------------------------------
    # Run complete AI pipeline
    # --------------------------------------------------------

    try:

        result = screening_service.screen(
            image=image,
            user_id=user_id,
            db=db,
            language=norm_lang,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        # Make sure the database transaction is
        # not left open after an unexpected error.

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Screening failed: {str(exc)}"
            ),
        )

    # --------------------------------------------------------
    # Localize screening result and generate narrative
    # --------------------------------------------------------

    localized_screening = localize_screening_response(
        result,
        language=norm_lang,
    )

    ai_narrative = gemini_service.generate_report_narrative(
        screening_data=result,
        language=norm_lang,
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "filename":
            file.filename,

        "user_id":
            user_id,

        "language":
            norm_lang,

        "screening_status":
            result.get(
                "status",
                "completed",
            ),

        "quality":
            localized_screening.get(
                "quality"
            ),

        "screening":
            localized_screening,

        "ai_report":
            ai_narrative,
    }