from collections import Counter


class EvidenceService:
    """
    Builds structured AI evidence from:

    1. DR classification
    2. Image quality
    3. Lesion detection

    This service currently works locally.

    Later, this structured evidence can be passed
    to a decision/reasoning layer such as Jev.
    """

    def build_evidence(
        self,
        dr_result,
        quality_result,
        lesion_result
    ):

        # =====================================================
        # LESION INFORMATION
        # =====================================================

        lesions = lesion_result.get(
            "lesions",
            []
        )

        # Count lesion types

        lesion_counts = Counter(
            lesion.get("type")
            for lesion in lesions
        )

        # =====================================================
        # STRUCTURED EVIDENCE
        # =====================================================

        evidence = {

            # -------------------------------------------------
            # DR CLASSIFICATION
            # -------------------------------------------------

            "dr_classification": {

                "grade":
                    dr_result.get("grade"),

                "label":
                    dr_result.get("label"),

                "confidence":
                    dr_result.get("confidence"),

                "probabilities":
                    dr_result.get(
                        "probabilities",
                        {}
                    )
            },

            # -------------------------------------------------
            # IMAGE QUALITY
            # -------------------------------------------------

            "image_quality": {

                "status":
                    quality_result.get(
                        "status"
                    ),

                "score":
                    quality_result.get(
                        "score"
                    ),

                "checks":
                    quality_result.get(
                        "checks",
                        {}
                    )
            },

            # -------------------------------------------------
            # LESION SUMMARY
            # -------------------------------------------------

            "lesion_evidence": {

                "total_lesions":
                    len(lesions),

                "microaneurysms":
                    lesion_counts.get(
                        "microaneurysm",
                        0
                    ),

                "hemorrhages":
                    lesion_counts.get(
                        "hemorrhage",
                        0
                    ),

                "hard_exudates":
                    lesion_counts.get(
                        "hard_exudate",
                        0
                    ),

                "soft_exudates":
                    lesion_counts.get(
                        "soft_exudate",
                        0
                    )
            },

            # -------------------------------------------------
            # DETAILED LESIONS
            # -------------------------------------------------

            "lesion_details":
                lesions
        }

        return evidence