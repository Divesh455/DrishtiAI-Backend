from pathlib import Path
from datetime import datetime
import uuid


# =============================================================
# PROJECT ROOT
# =============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = (
    BASE_DIR /
    "backend" /
    "outputs"
)


class OutputService:

    """
    Manages unique output folders for every screening.

    Structure:

        backend/
        └── outputs/
            └── screenings/
                └── screening_YYYYMMDD_HHMMSS_xxxx/
                    ├── heatmaps/
                    ├── overlays/
                    └── reports/
    """

    def __init__(self):

        self.screenings_dir = (
            OUTPUT_DIR /
            "screenings"
        )

        self.screenings_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================
    # CREATE UNIQUE SCREENING ID
    # =========================================================

    def create_screening_id(self):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        unique_part = uuid.uuid4().hex[:4]

        screening_id = (
            f"screening_{timestamp}_{unique_part}"
        )

        return screening_id

    # =========================================================
    # CREATE SCREENING FOLDER
    # =========================================================

    def create_screening_folder(
        self,
        screening_id
    ):

        screening_dir = (
            self.screenings_dir /
            screening_id
        )

        heatmap_dir = (
            screening_dir /
            "heatmaps"
        )

        overlay_dir = (
            screening_dir /
            "overlays"
        )

        report_dir = (
            screening_dir /
            "reports"
        )

        # -----------------------------------------------------
        # Create directories
        # -----------------------------------------------------

        heatmap_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        overlay_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        report_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        return {

            "screening_dir":
                screening_dir,

            "heatmap_dir":
                heatmap_dir,

            "overlay_dir":
                overlay_dir,

            "report_dir":
                report_dir
        }