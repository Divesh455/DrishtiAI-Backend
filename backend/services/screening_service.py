import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch

import cloudinary
import cloudinary.uploader

from dotenv import load_dotenv
from sqlalchemy.orm import Session

try:
    import onnxruntime as ort
except ImportError as exc:
    raise RuntimeError(
        "onnxruntime is required for the Senanur safety model. "
        "Install it with: pip install onnxruntime"
    ) from exc


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
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(
    BASE_DIR / "backend" / ".env"
)


# ============================================================
# CLOUDINARY CONFIGURATION
# ============================================================

CLOUDINARY_CLOUD_NAME = os.getenv(
    "CLOUDINARY_CLOUD_NAME"
)

CLOUDINARY_API_KEY = os.getenv(
    "CLOUDINARY_API_KEY"
)

CLOUDINARY_API_SECRET = os.getenv(
    "CLOUDINARY_API_SECRET"
)

if not all(
    [
        CLOUDINARY_CLOUD_NAME,
        CLOUDINARY_API_KEY,
        CLOUDINARY_API_SECRET,
    ]
):
    raise RuntimeError(
        "Cloudinary configuration is missing. "
        "Please set CLOUDINARY_CLOUD_NAME, "
        "CLOUDINARY_API_KEY and "
        "CLOUDINARY_API_SECRET in backend/.env"
    )

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True,
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from backend.models.dr_classifier import create_model

from backend.services.quality_service import (
    QualityService
)

from backend.models.lesion_detector import (
    LesionDetector
)

from backend.services.jev_service import (
    EvidenceService
)

from backend.services.output_service import (
    OutputService
)

from backend.xai.gradcam import (
    GradCAMExplainer
)

from backend.database.models import (
    Screening
)

from training.config import (
    DEVICE,
    NUM_CLASSES,
    IMAGE_SIZE,
    CLASS_NAMES,
)


# ============================================================
# MODEL PATHS
# ============================================================

PRIMARY_DR_MODEL_PATH = (
    BASE_DIR
    / "weights"
    / "dr_model_clean.pth"
)

SENANUR_MODEL_DIR = (
    BASE_DIR
    / "external_models"
    / "senanur_dr"
)

LESION_MODEL_DEFAULT_PATH = (
    BASE_DIR
    / "weights"
    / "lesion_model.pth"
)


# Senanur thresholds validated on the clean APTOS
# validation set.
SENANUR_THRESHOLDS = [
    0.668,
    1.132,
    2.324,
    3.3,
]

SENANUR_INPUT_SIZE = 384


# ============================================================
# SCREENING SERVICE
# ============================================================

class ScreeningService:

    """
    Complete DrishtiAI screening pipeline.

    Primary model:
        DrishtiAI clean EfficientNet-B0

    Secondary safety model:
        Senanur 5-fold ONNX ensemble

    Lesion model:
        ClementP fundus-lesions-toolkit via LesionDetector

    Decision logic:
        1. Quality Checker
        2. DrishtiAI Grade 0-4
        3. Grade >= 2 -> REFER
        4. Grade 0/1 -> Senanur safety check
        5. Senanur Grade >= 2 -> REFER
        6. Otherwise -> NON-REFERABLE

    Visual evidence:
        Grad-CAM++ and ClementP lesion overlay are generated
        for every quality-passing screening.

    Existing Cloudinary upload and Neon storage flow is preserved.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        dr_model_path=PRIMARY_DR_MODEL_PATH,
        senanur_model_dir=SENANUR_MODEL_DIR,
        lesion_model_path=LESION_MODEL_DEFAULT_PATH,
    ):

        self.device = DEVICE

        print(
            "\nInitializing "
            "DrishtiAI Screening Service..."
        )

        # ====================================================
        # DR CLASSIFIER
        # ====================================================

        print(
            "\nLoading DR classification model..."
        )

        print(
            f"DR model path: {dr_model_path}"
        )

        if not Path(dr_model_path).exists():
            raise FileNotFoundError(
                "DrishtiAI model not found: "
                f"{dr_model_path}"
            )

        self.dr_model = create_model(
            num_classes=NUM_CLASSES,
            device=self.device,
        )

        checkpoint = torch.load(
            dr_model_path,
            map_location=self.device,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint

        # dr_model_clean.pth was trained from the underlying
        # EfficientNet model, while DRClassifier wraps it as
        # self.model.
        target_model = (
            self.dr_model.model
            if hasattr(self.dr_model, "model")
            else self.dr_model
        )

        # Remove a DataParallel prefix if a checkpoint has one.
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[7:]

            cleaned_state_dict[new_key] = value

        target_model.load_state_dict(
            cleaned_state_dict
        )

        self.dr_model.to(
            self.device
        )

        self.dr_model.eval()

        print(
            "DR classification model loaded."
        )

        # ====================================================
        # SENANUR
        # ====================================================

        print(
            "\nLoading Senanur safety model..."
        )

        self.senanur_model_dir = Path(
            senanur_model_dir
        )

        self.senanur_sessions = (
            self._load_senanur_models(
                self.senanur_model_dir
            )
        )

        print(
            "Senanur safety model loaded "
            "(5 ONNX folds)."
        )

        # ====================================================
        # GRAD-CAM++
        # ====================================================

        print(
            "\nLoading Grad-CAM++ explainer..."
        )

        self.gradcam = GradCAMExplainer(
            model=self.dr_model,
            device=self.device,
            image_size=IMAGE_SIZE,
        )

        print(
            "Grad-CAM++ explainer loaded."
        )

        # ====================================================
        # IMAGE QUALITY
        # ====================================================

        print(
            "\nLoading image quality service..."
        )

        self.quality_service = (
            QualityService()
        )

        print(
            "Image quality service loaded."
        )

        # ====================================================
        # LESION DETECTOR
        # ====================================================

        print(
            "\nLoading lesion detection model..."
        )

        # LesionDetector now uses ClementP internally.
        # The existing model_path argument is retained for
        # compatibility with the current service interface.
        self.lesion_detector = (
            LesionDetector(
                model_path=str(
                    lesion_model_path
                ),
                image_size=512,
                device=self.device,
                confidence_threshold=0.40,
            )
        )

        print(
            "ClementP lesion detector loaded."
        )

        # ====================================================
        # EVIDENCE SERVICE
        # ====================================================

        print(
            "\nLoading evidence service..."
        )

        self.evidence_service = (
            EvidenceService()
        )

        print(
            "Evidence service loaded."
        )

        # ====================================================
        # OUTPUT SERVICE
        # ====================================================

        print(
            "\nLoading output service..."
        )

        self.output_service = (
            OutputService()
        )

        print(
            "Output service loaded."
        )

        print(
            "\nScreening service ready."
        )

    # ========================================================
    # LOAD SENANUR MODELS
    # ========================================================

    @staticmethod
    def _load_senanur_models(
        model_dir
    ):

        model_dir = Path(
            model_dir
        )

        if not model_dir.exists():
            raise FileNotFoundError(
                "Senanur model directory not found: "
                f"{model_dir}"
            )

        sessions = []

        for fold in range(1, 6):

            model_path = (
                model_dir
                / f"fold{fold}.onnx"
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Senanur fold {fold} not found: "
                    f"{model_path}"
                )

            session = ort.InferenceSession(
                str(model_path),
                providers=[
                    "CPUExecutionProvider"
                ],
            )

            inputs = session.get_inputs()

            outputs = session.get_outputs()

            if len(inputs) != 1:
                raise RuntimeError(
                    "Unexpected Senanur input count "
                    f"in fold {fold}: {len(inputs)}"
                )

            if len(outputs) < 1:
                raise RuntimeError(
                    "Senanur fold returned no outputs: "
                    f"{fold}"
                )

            input_shape = list(
                inputs[0].shape
            )

            if (
                len(input_shape) != 4
                or input_shape[1:] != [
                    3,
                    SENANUR_INPUT_SIZE,
                    SENANUR_INPUT_SIZE,
                ]
            ):
                raise RuntimeError(
                    "Unexpected Senanur input shape "
                    f"for fold {fold}: "
                    f"{input_shape}"
                )

            sessions.append(
                session
            )

            print(
                f"Senanur fold {fold} loaded: "
                f"{model_path.name}"
            )

        return sessions

    # ========================================================
    # CONVERT TO RGB NUMPY
    # ========================================================

    def _to_rgb_numpy(
        self,
        image,
    ):

        """
        Convert PIL Image or OpenCV image
        to RGB NumPy format.
        """

        if isinstance(
            image,
            Image.Image,
        ):

            image = image.convert("RGB")

            return np.array(
                image
            )

        if isinstance(
            image,
            np.ndarray,
        ):

            if image.ndim == 3:

                # The service passes OpenCV images as BGR.
                return cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

            if image.ndim == 2:

                return image

        raise TypeError(
            "Image must be a PIL Image "
            "or NumPy array."
        )

    # ========================================================
    # DR CLASSIFIER PREPROCESSING
    # ========================================================

    def _preprocess_for_classifier(
        self,
        image,
    ):

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        pil_image = (
            Image.fromarray(
                rgb_image
            ).convert("RGB")
        )

        pil_image = pil_image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        image_array = (
            np.array(
                pil_image
            ).astype(
                np.float32
            ) / 255.0
        )

        mean = np.array(
            [
                0.485,
                0.456,
                0.406,
            ],
            dtype=np.float32,
        )

        std = np.array(
            [
                0.229,
                0.224,
                0.225,
            ],
            dtype=np.float32,
        )

        image_array = (
            image_array - mean
        ) / std

        image_array = np.transpose(
            image_array,
            (2, 0, 1),
        )

        tensor = torch.tensor(
            image_array,
            dtype=torch.float32,
        )

        tensor = tensor.unsqueeze(0)

        return tensor.to(
            self.device
        )

    # ========================================================
    # DR CLASSIFICATION
    # ========================================================

    def _classify_dr(
        self,
        image,
    ):

        input_tensor = (
            self._preprocess_for_classifier(
                image
            )
        )

        with torch.no_grad():

            output = self.dr_model(
                input_tensor
            )

            probabilities = (
                torch.softmax(
                    output,
                    dim=1,
                )
            )

            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1,
                )
            )

        grade = int(
            prediction.item()
        )

        confidence = float(
            confidence.item()
        )

        probability_values = (
            probabilities[0]
            .cpu()
            .numpy()
            .tolist()
        )

        probability_dict = {}

        for index in range(
            NUM_CLASSES
        ):

            probability_dict[
                CLASS_NAMES[index]
            ] = round(
                float(
                    probability_values[
                        index
                    ]
                ),
                4,
            )

        return {
            "grade": grade,
            "label": CLASS_NAMES[grade],
            "confidence": round(
                confidence,
                4,
            ),
            "probabilities":
                probability_dict,
        }

    # ========================================================
    # SENANUR PREPROCESSING
    # ========================================================

    @staticmethod
    def _crop_image_from_gray(
        image,
        tol=7,
    ):

        """
        Remove dark borders using the preprocessing that was
        validated with the Senanur APTOS model.
        """

        if image is None:
            raise ValueError(
                "Image is None."
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        mask = gray > tol

        if not np.any(mask):
            return image

        rows = np.where(
            mask.any(axis=1)
        )[0]

        cols = np.where(
            mask.any(axis=0)
        )[0]

        if (
            len(rows) == 0
            or len(cols) == 0
        ):
            return image

        return image[
            rows[0]:rows[-1] + 1,
            cols[0]:cols[-1] + 1,
        ]

    def _preprocess_for_senanur(
        self,
        image,
    ):

        """
        Senanur preprocessing:

        1. RGB -> BGR for OpenCV
        2. Auto crop, tolerance 7
        3. Fit inside 512x512
        4. Zero pad to 512x512
        5. Resize to 384x384
        6. BGR -> RGB
        7. /255
        8. ImageNet normalization
        9. HWC -> CHW
        10. Batch dimension
        """

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        if rgb_image.ndim != 3:
            raise ValueError(
                "Senanur requires a color fundus image."
            )

        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        bgr_image = (
            self._crop_image_from_gray(
                bgr_image,
                tol=7,
            )
        )

        if bgr_image.size == 0:
            raise ValueError(
                "Senanur preprocessing produced "
                "an empty image."
            )

        h, w = bgr_image.shape[:2]

        scale = (
            512.0
            / max(h, w)
        )

        new_w = max(
            1,
            int(round(w * scale))
        )

        new_h = max(
            1,
            int(round(h * scale))
        )

        bgr_image = cv2.resize(
            bgr_image,
            (
                new_w,
                new_h,
            ),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.zeros(
            (
                512,
                512,
                3,
            ),
            dtype=np.uint8,
        )

        y_offset = (
            512 - new_h
        ) // 2

        x_offset = (
            512 - new_w
        ) // 2

        canvas[
            y_offset:
            y_offset + new_h,
            x_offset:
            x_offset + new_w,
        ] = bgr_image

        bgr_image = cv2.resize(
            canvas,
            (
                SENANUR_INPUT_SIZE,
                SENANUR_INPUT_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        rgb_image = cv2.cvtColor(
            bgr_image,
            cv2.COLOR_BGR2RGB,
        )

        image_array = (
            rgb_image.astype(
                np.float32
            ) / 255.0
        )

        mean = np.array(
            [
                0.485,
                0.456,
                0.406,
            ],
            dtype=np.float32,
        )

        std = np.array(
            [
                0.229,
                0.224,
                0.225,
            ],
            dtype=np.float32,
        )

        image_array = (
            image_array - mean
        ) / std

        image_array = np.transpose(
            image_array,
            (2, 0, 1),
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        return image_array.astype(
            np.float32
        )

    # ========================================================
    # SENANUR SCORE -> GRADE
    # ========================================================

    @staticmethod
    def _senanur_score_to_grade(
        score,
    ):

        if score < SENANUR_THRESHOLDS[0]:
            return 0

        if score < SENANUR_THRESHOLDS[1]:
            return 1

        if score < SENANUR_THRESHOLDS[2]:
            return 2

        if score < SENANUR_THRESHOLDS[3]:
            return 3

        return 4

    # ========================================================
    # SENANUR SAFETY CHECK
    # ========================================================

    def _check_senanur_safety(
        self,
        image,
    ):

        """
        Run all five Senanur folds and average their ordinal
        severity scores.

        This method is intentionally called only when the
        primary DrishtiAI prediction is Grade 0 or Grade 1.
        """

        input_array = (
            self._preprocess_for_senanur(
                image
            )
        )

        fold_scores = []

        for session in (
            self.senanur_sessions
        ):

            input_name = (
                session
                .get_inputs()[0]
                .name
            )

            outputs = session.run(
                None,
                {
                    input_name:
                        input_array
                },
            )

            if not outputs:
                raise RuntimeError(
                    "Senanur returned no output."
                )

            raw_output = np.asarray(
                outputs[0]
            ).reshape(-1)

            if raw_output.size != 1:
                raise RuntimeError(
                    "Unexpected Senanur output shape: "
                    f"{raw_output.shape}"
                )

            fold_scores.append(
                float(
                    raw_output[0]
                )
            )

        fold_scores = np.asarray(
            fold_scores,
            dtype=np.float32
        )

        mean_score = float(
            np.mean(
                fold_scores
            )
        )

        fold_spread = float(
            np.std(
                fold_scores
            )
        )

        senanur_grade = (
            self._senanur_score_to_grade(
                mean_score
            )
        )

        referable = (
            senanur_grade >= 2
        )

        return {
            "enabled": True,

            "model": (
                "Senanur 5-fold "
                "EfficientNet-B0 ONNX ensemble"
            ),

            "raw_score":
                round(
                    mean_score,
                    4,
                ),

            "grade":
                int(
                    senanur_grade
                ),

            "label":
                CLASS_NAMES[
                    senanur_grade
                ],

            "referable":
                bool(
                    referable
                ),

            "fold_spread":
                round(
                    fold_spread,
                    4,
                ),

            "fold_scores": [
                round(
                    float(score),
                    4,
                )
                for score in fold_scores
            ],

            "thresholds":
                SENANUR_THRESHOLDS,
        }

    # ========================================================
    # FINAL SCREENING DECISION
    # ========================================================

    def _build_final_decision(
        self,
        image,
        classification_result,
    ):

        """
        Apply the locked DrishtiAI + Senanur screening rule.
        """

        dr_grade = int(
            classification_result[
                "grade"
            ]
        )

        # ----------------------------------------------------
        # Primary model already says referable.
        # ----------------------------------------------------

        if dr_grade >= 2:

            return {
                "decision":
                    "REFER",

                "referable":
                    True,

                "reason": (
                    "DrishtiAI predicted "
                    f"Grade {dr_grade}, which is "
                    "referable (Grade >= 2)."
                ),

                "senanur_used":
                    False,

                "senanur":
                    None,
            }

        # ----------------------------------------------------
        # DrishtiAI Grade 0/1 -> Senanur safety check.
        # ----------------------------------------------------

        senanur_result = (
            self._check_senanur_safety(
                image
            )
        )

        if senanur_result["referable"]:

            return {
                "decision":
                    "REFER",

                "referable":
                    True,

                "reason": (
                    "DrishtiAI predicted Grade "
                    f"{dr_grade}, but the Senanur "
                    "safety check predicted "
                    f"Grade {senanur_result['grade']}."
                ),

                "senanur_used":
                    True,

                "senanur":
                    senanur_result,
            }

        return {
            "decision":
                "NON-REFERABLE",

            "referable":
                False,

            "reason": (
                "DrishtiAI predicted Grade "
                f"{dr_grade} and the Senanur "
                "safety check did not indicate "
                "referable DR."
            ),

            "senanur_used":
                True,

            "senanur":
                senanur_result,
        }

    # ========================================================
    # IMAGE QUALITY
    # ========================================================

    def _check_quality(
        self,
        image,
    ):

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        pil_image = Image.fromarray(
            rgb_image
        ).convert("RGB")

        return (
            self.quality_service.check_pil_image(
                pil_image
            )
        )

    # ========================================================
    # GRAD-CAM++
    # ========================================================

    def _generate_gradcam(
        self,
        image,
        target_class,
    ):

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        pil_image = (
            Image.fromarray(
                rgb_image
            ).convert("RGB")
        )

        return (
            self.gradcam.generate(
                image=pil_image,
                target_class=target_class,
            )
        )

    # ========================================================
    # UPLOAD GRAD-CAM RESULTS
    # ========================================================

    def _upload_gradcam(
        self,
        gradcam_result,
        screening_id,
    ):

        return (
            self.gradcam.upload_results(
                result=gradcam_result,
                screening_id=screening_id,
            )
        )

    # ========================================================
    # LESION DETECTION
    # ========================================================

    def _detect_lesions(
        self,
        image,
        screening_id,
    ):

        """
        Run ClementP lesion segmentation.

        The LesionDetector handles inference, visualization,
        and Cloudinary overlay upload.
        """

        rgb_image = (
            self._to_rgb_numpy(
                image
            )
        )

        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        return (
            self.lesion_detector.detect(
                bgr_image,
                screening_id=screening_id,
            )
        )

    # ========================================================
    # BUILD EVIDENCE
    # ========================================================

    def _build_evidence(
        self,
        classification_result,
        quality_result,
        lesion_result,
    ):

        result = (
            self.evidence_service.build_evidence(
                dr_result=classification_result,
                quality_result=quality_result,
                lesion_result=lesion_result,
            )
        )

        if result is None:
            result = {}

        return result

    # ========================================================
    # SAVE SCREENING TO NEON
    # ========================================================

    def _save_screening(
        self,
        db: Session,
        user_id: str,
        screening_id: str,
        quality_result,
        classification_result,
        lesion_result,
        evidence,
        images,
        screening_status="completed",
    ):

        """
        Save complete screening result to Neon PostgreSQL.

        IMPORTANT:
        The existing Screening schema and field mapping are
        preserved. The final referral decision is stored inside
        the existing JSON evidence field.
        """

        quality_score = (
            quality_result.get(
                "score"
            )
        )

        quality_status = (
            quality_result.get(
                "status"
            )
        )

        fundus_data = (
            quality_result.get(
                "fundus_suitability"
            )
            if quality_result
            else None
        )

        if isinstance(fundus_data, dict):
            fundus_suitability = fundus_data.get("is_fundus")
            fundus_confidence = fundus_data.get("confidence")
        else:
            fundus_suitability = fundus_data
            fundus_confidence = (
                quality_result.get("fundus_confidence")
                if quality_result
                else None
            )

        screening = Screening(

            screening_id=
                screening_id,

            user_id=
                str(user_id),

            dr_grade=(
                classification_result.get(
                    "grade"
                )
                if classification_result
                else None
            ),

            dr_label=(
                classification_result.get(
                    "label"
                )
                if classification_result
                else None
            ),

            dr_confidence=(
                classification_result.get(
                    "confidence"
                )
                if classification_result
                else None
            ),

            class_probabilities=(
                classification_result.get(
                    "probabilities"
                )
                if classification_result
                else None
            ),

            quality_score=
                quality_score,

            quality_status=
                quality_status,

            fundus_suitability=
                fundus_suitability,

            fundus_confidence=
                fundus_confidence,

            lesion_count=(
                lesion_result.get(
                    "lesion_count",
                    0,
                )
                if lesion_result
                else 0
            ),

            lesion_details=(
                lesion_result.get(
                    "lesions",
                    lesion_result.get(
                        "lesion_details"
                    ),
                )
                if lesion_result
                else None
            ),

            evidence=
                evidence,

            original_image_url=(
                images.get(
                    "original_url"
                )
                if images
                else None
            ),

            heatmap_url=(
                images.get(
                    "heatmap_url"
                )
                if images
                else None
            ),

            gradcam_url=(
                images.get(
                    "gradcam_url"
                )
                if images
                else None
            ),

            lesion_overlay_url=(
                images.get(
                    "lesion_overlay_url"
                )
                if images
                else None
            ),

            screening_status=
                screening_status,
        )

        try:

            db.add(
                screening
            )

            db.commit()

            db.refresh(
                screening
            )

            print(
                f"\nScreening saved to Neon: "
                f"{screening_id}"
            )

            return screening

        except Exception:

            db.rollback()

            print(
                "\nFailed to save screening "
                "to Neon."
            )

            raise

    # ========================================================
    # COMPLETE SCREENING
    # ========================================================

    def screen(
        self,
        image,
        user_id: str,
        db: Session,
        screening_id=None,
    ):

        """
        Run complete screening and save the result to Neon.
        """

        # ====================================================
        # VALIDATE USER ID
        # ====================================================

        if not user_id:

            raise ValueError(
                "user_id is required for screening."
            )

        # ====================================================
        # CREATE SCREENING ID
        # ====================================================

        if screening_id is None:

            screening_id = (
                self.output_service
                .create_screening_id()
            )

        print(
            f"\nScreening ID: {screening_id}"
        )

        print(
            f"User ID: {user_id}"
        )

        # ====================================================
        # STEP 1 — QUALITY
        # ====================================================

        print(
            "\n[1/5] Checking image quality..."
        )

        quality_result = (
            self._check_quality(
                image
            )
        )

        print(
            f"Quality status: "
            f"{quality_result.get('status')}"
        )

        print(
            f"Quality score: "
            f"{quality_result.get('score')}"
        )

        # ====================================================
        # STOP IF POOR
        # ====================================================

        if (
            quality_result.get(
                "status"
            )
            == "poor"
        ):

            print(
                "\nImage rejected because "
                "quality is insufficient."
            )

            rejected_result = {

                "screening_id":
                    screening_id,

                "user_id":
                    str(user_id),

                "status":
                    "rejected",

                "message": (
                    "Image quality is "
                    "insufficient for reliable "
                    "AI screening."
                ),

                "quality":
                    quality_result,

                "classification":
                    None,

                "screening_decision":
                    None,

                "explainability":
                    None,

                "lesions":
                    None,

                "evidence":
                    None,

                "images":
                    None,
            }

            self._save_screening(

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

            return rejected_result

        # ====================================================
        # STEP 2 — DR CLASSIFICATION + SENANUR SAFETY
        # ====================================================

        print(
            "\n[2/5] Running DR classification..."
        )

        classification_result = (
            self._classify_dr(
                image
            )
        )

        print(
            f"DR Grade: "
            f"{classification_result['grade']}"
        )

        print(
            f"DR Label: "
            f"{classification_result['label']}"
        )

        print(
            f"Confidence: "
            f"{classification_result['confidence']}"
        )

        screening_decision = (
            self._build_final_decision(
                image=image,
                classification_result=(
                    classification_result
                ),
            )
        )

        print(
            f"Screening decision: "
            f"{screening_decision['decision']}"
        )

        if screening_decision.get(
            "senanur_used"
        ):

            senanur_result = (
                screening_decision.get(
                    "senanur"
                )
                or {}
            )

            print(
                "Senanur Grade: "
                f"{senanur_result.get('grade')}"
            )

            print(
                "Senanur Referable: "
                f"{senanur_result.get('referable')}"
            )

        # ====================================================
        # STEP 3 — GRAD-CAM++
        # ====================================================

        print(
            "\n[3/5] Generating "
            "Grad-CAM++ explanation..."
        )

        gradcam_result = (
            self._generate_gradcam(
                image=image,
                target_class=(
                    classification_result[
                        "grade"
                    ]
                ),
            )
        )

        print(
            "Grad-CAM++ explanation generated."
        )

        print(
            "Uploading Grad-CAM images "
            "to Cloudinary..."
        )

        gradcam_urls = (
            self._upload_gradcam(
                gradcam_result=gradcam_result,
                screening_id=screening_id,
            )
        )

        print(
            "Grad-CAM images uploaded."
        )

        # ====================================================
        # STEP 4 — CLEMENTP LESION DETECTION
        # ====================================================

        print(
            "\n[4/5] Running ClementP lesion detection..."
        )

        lesion_result = (
            self._detect_lesions(
                image=image,
                screening_id=screening_id,
            )
        )

        print(
            f"Lesion count: "
            f"{lesion_result.get('lesion_count', 0)}"
        )

        if lesion_result.get(
            "overlay_url"
        ):

            print(
                "ClementP lesion overlay uploaded "
                "to Cloudinary."
            )

        # ====================================================
        # STEP 5 — EVIDENCE
        # ====================================================

        print(
            "\n[5/5] Building structured evidence..."
        )

        evidence = (
            self._build_evidence(
                classification_result=(
                    classification_result
                ),
                quality_result=(
                    quality_result
                ),
                lesion_result=(
                    lesion_result
                ),
            )
        )

        # Store the final screening decision in the existing
        # Neon JSON evidence column. No schema change.
        evidence[
            "screening_decision"
        ] = screening_decision

        print(
            "Structured evidence created."
        )

        # ====================================================
        # CLOUDINARY IMAGE INFORMATION
        # ====================================================

        images = {

            "original_url": (
                gradcam_urls.get(
                    "original_url"
                )
            ),

            "heatmap_url": (
                gradcam_urls.get(
                    "heatmap_url"
                )
            ),

            "gradcam_url": (
                gradcam_urls.get(
                    "overlay_url"
                )
            ),

            "lesion_overlay_url": (
                lesion_result.get(
                    "overlay_url"
                )
            ),
        }

        # ====================================================
        # SAVE COMPLETE RESULT TO NEON
        # ====================================================

        print(
            "\nSaving complete screening "
            "result to Neon..."
        )

        self._save_screening(

            db=db,

            user_id=user_id,

            screening_id=screening_id,

            quality_result=quality_result,

            classification_result=(
                classification_result
            ),

            lesion_result=lesion_result,

            evidence=evidence,

            images=images,

            screening_status="completed",
        )

        # ====================================================
        # FINAL EXPLAINABILITY RESULT
        # ====================================================

        explainability = {

            "method":
                "Grad-CAM++",

            "target_class":
                gradcam_result.get(
                    "target_class"
                ),

            "predicted_class":
                gradcam_result.get(
                    "predicted_class"
                ),

            "confidence":
                classification_result.get(
                    "confidence"
                ),

            "explanation": (
                "Grad-CAM++ highlights "
                "image regions that contributed "
                "to the selected DR classification. "
                "The highlighted regions represent "
                "model attention and should not be "
                "interpreted as individual lesions "
                "without supporting lesion detection "
                "evidence."
            ),

            "heatmap_url":
                gradcam_urls.get(
                    "heatmap_url"
                ),

            "overlay_url":
                gradcam_urls.get(
                    "overlay_url"
                ),
        }

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {

            "screening_id":
                screening_id,

            "user_id":
                str(user_id),

            "status":
                "completed",

            "quality":
                quality_result,

            "classification":
                classification_result,

            "screening_decision":
                screening_decision,

            "explainability":
                explainability,

            "lesions":
                lesion_result,

            "evidence":
                evidence,

            "images":
                images,
        }


# ============================================================
# OPTIONAL LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Neon + Cloudinary enabled "
        "ScreeningService module loaded."
    )
