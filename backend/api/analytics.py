from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, inspect
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from backend.database.database import get_db
from backend.database.models import Screening

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Analytics"],
)


@router.get("/overview")
def get_analytics_overview(
    days: Optional[int] = Query(
        default=None,
        ge=1,
        le=3650,
        description="Filter stats for the last N days (e.g., 7, 30, 365). Default is all-time.",
    ),
    user_id: Optional[str] = Query(
        default=None,
        description="Optional filter for a specific patient/user ID.",
    ),
    db: Session = Depends(get_db),
):
    """
    Returns population-wide screening metrics, referral decision rates,
    DR grade distribution, and quality assessment statistics for doctors and administrators.

    Examples:
        GET /api/v1/analytics/overview
        GET /api/v1/analytics/overview?days=30
        GET /api/v1/analytics/overview?user_id=USER_9876
    """

    try:
        query = db.query(Screening)

        if user_id:
            query = query.filter(Screening.user_id == str(user_id))

        if days and days > 0:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            query = query.filter(Screening.created_at >= cutoff_date)

        all_records = query.all()

        total_screenings = len(all_records)
        completed_records = [r for r in all_records if r.screening_status == "completed"]
        rejected_records = [r for r in all_records if r.screening_status == "rejected"]

        unique_user_ids = {r.user_id for r in all_records if r.user_id}

        # DR Grade Breakdown
        grade_0 = sum(1 for r in completed_records if r.dr_grade == 0)
        grade_1 = sum(1 for r in completed_records if r.dr_grade == 1)
        grade_2 = sum(1 for r in completed_records if r.dr_grade == 2)
        grade_3 = sum(1 for r in completed_records if r.dr_grade == 3)
        grade_4 = sum(1 for r in completed_records if r.dr_grade == 4)
        unclassified = sum(1 for r in completed_records if r.dr_grade is None)

        # Referral Decision Breakdown (Grade >= 2 is REFER)
        refer_count = sum(1 for r in completed_records if r.dr_grade is not None and r.dr_grade >= 2)
        non_referable_count = sum(1 for r in completed_records if r.dr_grade is not None and r.dr_grade < 2)

        total_classified = refer_count + non_referable_count
        referral_percentage = round((refer_count / total_classified * 100), 2) if total_classified > 0 else 0.0

        # Quality Assessment Breakdown
        good_quality = sum(1 for r in all_records if (r.quality_status or "").lower() == "good")
        adequate_quality = sum(1 for r in all_records if (r.quality_status or "").lower() == "adequate")
        poor_quality = sum(1 for r in all_records if (r.quality_status or "").lower() in ["poor", "rejected"])

        # Lesions accumulated
        total_lesions = sum(r.lesion_count or 0 for r in completed_records)

        return {
            "status": "success",
            "timeframe_days": days if days else "all_time",
            "filter_user_id": user_id,
            "overview": {
                "total_screenings": total_screenings,
                "unique_patients": len(unique_user_ids),
                "completed_screenings": len(completed_records),
                "rejected_screenings": len(rejected_records),
            },
            "referral_decisions": {
                "refer_count": refer_count,
                "non_referable_count": non_referable_count,
                "total_evaluated": total_classified,
                "referral_percentage": referral_percentage,
            },
            "dr_grade_distribution": {
                "grade_0_no_dr": grade_0,
                "grade_1_mild": grade_1,
                "grade_2_moderate": grade_2,
                "grade_3_severe": grade_3,
                "grade_4_proliferative": grade_4,
                "unclassified": unclassified,
            },
            "quality_assessment": {
                "good_quality": good_quality,
                "adequate_quality": adequate_quality,
                "poor_quality": poor_quality,
            },
            "total_lesions_detected": total_lesions,
        }

    except Exception as exc:
        print(f"[AnalyticsAPI] Query error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate analytics overview: {str(exc)}",
        )
