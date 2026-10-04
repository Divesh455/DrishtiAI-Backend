import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import torch

import cloudinary
import cloudinary.uploader

from dotenv import load_dotenv

from fundus_lesions_toolkit.models import segment
from fundus_lesions_toolkit.models.segmentation import get_model
from fundus_lesions_toolkit.constants import Dataset


# ============================================================
# PROJECT / CLOUDINARY CONFIGURATION
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

load_dotenv(
    BASE_DIR / "backend" / ".env"
)

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
# CLEMENTP LESION DEFINITIONS
# ============================================================

# ClementP current toolkit returns 5 channels:
# 0 = Background
# 1 = Cotton Wool Spots
# 2 = Exudates
# 3 = Hemorrhages
# 4 = Microaneurysms
#
# Source:
# https://github.com/ClementPla/fundus-lesions-toolkit
#
# NOTE:
# The toolkit does NOT separate hard vs soft exudates.
# Therefore this detector reports the toolkit's "exudates"
# class as a single category.

CLASS_NAMES: Dict[int, str] = {
    1: "cotton_wool_spot",
    2: "exudate",
    3: "hemorrhage",
    4: "microaneurysm",
}

CLASS_DISPLAY_NAMES: Dict[int, str] = {
    1: "Cotton Wool Spot",
    2: "Exudate",
    3: "Hemorrhage",
    4: "Microaneurysm",
}

# BGR colors for visualization.
# Kept distinct so the four lesion types are easy to read.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (255, 0, 255),  # Magenta
    2: (0, 255, 255),  # Yellow
    3: (0, 0, 255),  # Red
    4: (255, 0, 0),  # Blue
}

# This is an inference threshold, not a validated clinical threshold.
# It is intentionally kept configurable.
DEFAULT_CLASS_THRESHOLD = 0.50

MIN_AREAS: Dict[int, int] = {
    1: 8,
    2: 8,
    3: 8,
    4: 3,
}

MAX_AREA_RATIO: Dict[int, float] = {
    1: 0.20,
    2: 0.20,
    3: 0.20,
    4: 0.015,
}


# ============================================================
# LESION DETECTOR
# ============================================================


class LesionDetector:
    """
    ClementP fundus-lesions-toolkit based lesion segmentation.

    The detector preserves the existing DrishtiAI service interface:

        detector.detect(
            bgr_image,
            screening_id=...
        )

    Existing Cloudinary upload behavior is preserved:
        drishti-ai/screenings/{screening_id}/lesion_overlay

    The legacy local U-Net checkpoint is NOT loaded.

    ClementP's current toolkit provides four lesion classes:
        1. Cotton Wool Spots
        2. Exudates
        3. Hemorrhages
        4. Microaneurysms

    It returns a 5-channel post-softmax tensor:
        Background, CTW, EX, HE, MA
    """

    CLASS_NAMES = CLASS_NAMES
    CLASS_DISPLAY_NAMES = CLASS_DISPLAY_NAMES
    CLASS_COLORS = CLASS_COLORS
    CLASS_THRESHOLDS = {class_id: DEFAULT_CLASS_THRESHOLD for class_id in CLASS_NAMES}
    MIN_AREAS = MIN_AREAS
    MAX_AREA_RATIO = MAX_AREA_RATIO

    def __init__(
        self,
        model_path="weights/lesion_model.pth",
        image_size=512,
        device=None,
        confidence_threshold=0.40,
        train_datasets=Dataset.ALL,
        image_resolution=1024,
    ):
        """
        Parameters
        ----------
        model_path:
            Kept for backward compatibility with the existing
            ScreeningService. It is intentionally ignored because
            ClementP weights are loaded by the toolkit from
            Hugging Face.

        image_size:
            Kept for compatibility with the existing service.
            It is used as a fallback visualization/input target
            when needed, but ClementP's own preprocessing is used
            for inference.

        device:
            torch.device or None.

        confidence_threshold:
            Minimum probability threshold for lesion masks.

        train_datasets:
            ClementP model variant. Dataset.ALL uses the model
            trained on all supported training data.

        image_resolution:
            ClementP inference resolution before model processing.
        """

        self.legacy_model_path = Path(model_path) if model_path is not None else None

        self.image_size = int(image_size)

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.confidence_threshold = float(confidence_threshold)

        self.train_datasets = train_datasets

        self.image_resolution = int(image_resolution)

        self.model = None

        self._load_model()

    # ========================================================
    # LOAD CLEMENTP MODEL
    # ========================================================

    def _load_model(self):
        """
        Load and cache the ClementP pretrained segmentation model.

        The current ClementP toolkit uses:
            unet + seresnext50_32x4d
            Dataset.ALL by default.

        We deliberately do not load the legacy
        weights/lesion_model.pth file.
        """

        if self.legacy_model_path is not None:
            print(
                "[LesionDetector] Legacy lesion model path "
                f"ignored: {self.legacy_model_path}"
            )

        print("[LesionDetector] Loading ClementP lesion model...")

        try:
            self.model = get_model(
                arch="unet",
                encoder="seresnext50_32x4d",
                train_datasets=self.train_datasets,
                device=self.device,
                compile=False,
            )
        except TypeError:
            # Compatibility with an older installed toolkit where
            # get_model may expose a slightly different signature.
            self.model = get_model(
                arch="unet",
                encoder="seresnext50_32x4d",
                train_datasets=self.train_datasets,
                device=self.device,
            )

        self.model.to(self.device)

        self.model.eval()

        print("[LesionDetector] ClementP lesion model loaded.")

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_bgr_image(image):
        if image is None:
            raise ValueError("Input image is None.")

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError("Lesion detector expects a NumPy array.")

        if image.ndim != 3:
            raise ValueError("Lesion detector expects a 3-channel BGR image.")

        if image.shape[2] != 3:
            raise ValueError("Lesion detector expects exactly 3 channels.")

        if image.dtype != np.uint8:
            image = np.clip(
                image,
                0,
                255,
            ).astype(np.uint8)

        return image

    # ========================================================
    # SEGMENTATION
    # ========================================================

    def _segment(self, bgr_image):
        """
        Run ClementP lesion segmentation while bypassing the
        installed fundus_data_toolkit autocrop routine.

        The installed dependency currently fails inside
        autocrop_fundus() for this input. We therefore perform
        the resolution fitting here and tell ClementP not to
        run its automatic fitting.

        Input:
            BGR uint8 HxWx3

        Returns:
            5 x H x W probability maps at the original resolution.
        """

        bgr_image = self._validate_bgr_image(bgr_image)

        original_height, original_width = bgr_image.shape[:2]

        # --------------------------------------------------------
        # BGR -> RGB
        # --------------------------------------------------------

        rgb_image = cv2.cvtColor(
            bgr_image,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------------
        # Remove dark borders ourselves
        # --------------------------------------------------------

        gray = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2GRAY,
        )

        mask = gray > 7

        if np.any(mask):

            rows = np.where(mask.any(axis=1))[0]

            cols = np.where(mask.any(axis=0))[0]

            if len(rows) > 0 and len(cols) > 0:

                rgb_image = rgb_image[
                    rows[0] : rows[-1] + 1,
                    cols[0] : cols[-1] + 1,
                ]

        # --------------------------------------------------------
        # Fit longest dimension to 1024
        # --------------------------------------------------------

        height, width = rgb_image.shape[:2]

        scale = 1024.0 / max(height, width)

        resized_width = max(1, int(round(width * scale)))

        resized_height = max(1, int(round(height * scale)))

        fitted = cv2.resize(
            rgb_image,
            (
                resized_width,
                resized_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # --------------------------------------------------------
        # Pad to 1024 x 1024
        # --------------------------------------------------------

        canvas = np.zeros(
            (
                1024,
                1024,
                3,
            ),
            dtype=np.uint8,
        )

        y_offset = (1024 - resized_height) // 2

        x_offset = (1024 - resized_width) // 2

        canvas[
            y_offset : y_offset + resized_height,
            x_offset : x_offset + resized_width,
        ] = fitted

        # --------------------------------------------------------
        # ClementP segmentation
        #
        # IMPORTANT:
        # autofit_resolution=False prevents the dependency's
        # failing autocrop_fundus() code from running.
        # --------------------------------------------------------

        with torch.inference_mode():

            prediction = segment(
                canvas,
                arch="unet",
                encoder="seresnext50_32x4d",
                train_datasets=self.train_datasets,
                autofit_resolution=False,
                reverse_autofit=False,
                image_resolution=1024,
                device=self.device,
                compile=False,
            )

        if isinstance(
            prediction,
            tuple,
        ):

            prediction = prediction[0]

        if isinstance(
            prediction,
            torch.Tensor,
        ):

            prediction = prediction.detach().cpu().numpy()

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        if prediction.ndim != 3:

            raise RuntimeError(
                "Unexpected ClementP segmentation output shape: " f"{prediction.shape}"
            )

        if prediction.shape[0] != 5:

            raise RuntimeError(
                "Unexpected ClementP class count: "
                f"{prediction.shape[0]}; expected 5."
            )

        # --------------------------------------------------------
        # Remove the artificial padding.
        # --------------------------------------------------------

        prediction = prediction[
            :,
            y_offset : y_offset + resized_height,
            x_offset : x_offset + resized_width,
        ]

        # --------------------------------------------------------
        # Resize segmentation maps back to original image size.
        # --------------------------------------------------------

        output = np.zeros(
            (
                5,
                original_height,
                original_width,
            ),
            dtype=np.float32,
        )

        for class_id in range(5):

            output[class_id] = cv2.resize(
                prediction[class_id],
                (
                    original_width,
                    original_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        return output

    # ========================================================
    # RETINAL MASK
    # ========================================================

    @staticmethod
    def _create_retinal_mask(
        image,
    ):
        """
        Create a conservative fundus-region mask to suppress
        detections in black borders/background.
        """

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY,
        )

        mask = (gray > 10).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        clean = np.zeros_like(mask)

        if num_labels > 1:
            areas = stats[
                1:,
                cv2.CC_STAT_AREA,
            ]

            largest = 1 + int(np.argmax(areas))

            clean[labels == largest] = 255

        if clean.max() == 0:
            h, w = clean.shape

            center = (
                w // 2,
                h // 2,
            )

            radius = int(min(w, h) * 0.46)

            cv2.circle(
                clean,
                center,
                radius,
                255,
                -1,
            )

        return clean.astype(np.float32) / 255.0

    # ========================================================
    # EXTRACT REGIONS
    # ========================================================

    def _get_regions(
        self,
        probability_map,
        class_id,
        retinal_mask,
    ):
        """
        Convert one ClementP probability map into connected
        lesion regions.
        """

        threshold = max(
            self.confidence_threshold,
            self.CLASS_THRESHOLDS.get(
                class_id,
                self.confidence_threshold,
            ),
        )

        probability_map = probability_map * retinal_mask

        binary = (probability_map >= threshold).astype(np.uint8) * 255

        if class_id == 4:
            # Microaneurysms can be very small.
            kernel_size = 3
        else:
            kernel_size = 5

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

        (
            num_labels,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        regions = []

        image_area = binary.shape[0] * binary.shape[1]

        minimum_area = self.MIN_AREAS.get(
            class_id,
            5,
        )

        maximum_area = image_area * self.MAX_AREA_RATIO.get(
            class_id,
            0.20,
        )

        for region_id in range(
            1,
            num_labels,
        ):
            area = int(
                stats[
                    region_id,
                    cv2.CC_STAT_AREA,
                ]
            )

            if area < minimum_area:
                continue

            if area > maximum_area:
                continue

            region_mask = labels == region_id

            region_probabilities = probability_map[region_mask]

            if region_probabilities.size == 0:
                continue

            confidence = float(np.mean(region_probabilities))

            peak_confidence = float(np.max(region_probabilities))

            if confidence < threshold:
                continue

            x = int(
                stats[
                    region_id,
                    cv2.CC_STAT_LEFT,
                ]
            )

            y = int(
                stats[
                    region_id,
                    cv2.CC_STAT_TOP,
                ]
            )

            width = int(
                stats[
                    region_id,
                    cv2.CC_STAT_WIDTH,
                ]
            )

            height = int(
                stats[
                    region_id,
                    cv2.CC_STAT_HEIGHT,
                ]
            )

            center_x = float(
                centroids[
                    region_id,
                    0,
                ]
            )

            center_y = float(
                centroids[
                    region_id,
                    1,
                ]
            )

            region_binary = region_mask.astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                region_binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contour = None

            if contours:
                contour = max(
                    contours,
                    key=cv2.contourArea,
                )

            regions.append(
                {
                    "class_id": int(class_id),
                    "type": self.CLASS_NAMES[class_id],
                    "display_name": self.CLASS_DISPLAY_NAMES[class_id],
                    "bbox": [
                        x,
                        y,
                        x + width,
                        y + height,
                    ],
                    "area_pixels": area,
                    "area_ratio": float(area / image_area),
                    "center": [
                        round(
                            center_x,
                            2,
                        ),
                        round(
                            center_y,
                            2,
                        ),
                    ],
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "peak_confidence": round(
                        peak_confidence,
                        4,
                    ),
                    "_contour": contour,
                    "_mask": region_mask,
                }
            )

        return regions

    # ========================================================
    # DRAW LESION REGION
    # ========================================================

    def _draw_region(
        self,
        image,
        region,
        color,
        alpha=0.28,
    ):
        """
        Overlay the actual segmented region, not the full box.
        """

        result = image.copy()

        mask = region["_mask"]

        color_layer = np.zeros_like(image)

        color_layer[:] = color

        mask_float = mask.astype(np.float32)

        mask_float = mask_float * alpha

        mask_float = mask_float[..., None]

        result = (
            result.astype(np.float32) * (1.0 - mask_float)
            + color_layer.astype(np.float32) * mask_float
        )

        return np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    def create_overlay(
        self,
        image,
        lesions,
        alpha=0.28,
    ):
        """
        Create the Cloudinary-ready lesion overlay.
        """

        image = self._validate_bgr_image(image)

        result = image.copy()

        # ----------------------------------------------------
        # Filled regions
        # ----------------------------------------------------

        for lesion in lesions:

            class_id = int(lesion["class_id"])

            color = self.CLASS_COLORS.get(
                class_id,
                (255, 255, 255),
            )

            result = self._draw_region(
                result,
                lesion,
                color,
                alpha=alpha,
            )

        # ----------------------------------------------------
        # Contours, boxes, labels
        # ----------------------------------------------------

        for lesion in lesions:

            class_id = int(lesion["class_id"])

            color = self.CLASS_COLORS.get(
                class_id,
                (255, 255, 255),
            )

            x1, y1, x2, y2 = lesion["bbox"]

            contour = lesion.get("_contour")

            if contour is not None:

                cv2.drawContours(
                    result,
                    [contour],
                    -1,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.rectangle(
                result,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                color,
                1,
                cv2.LINE_AA,
            )

            label = f"{lesion['display_name']} " f"{lesion['confidence']:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.45
            thickness = 1

            (
                text_size,
                baseline,
            ) = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )

            text_width = text_size[0]

            text_height = text_size[1]

            label_x = max(
                2,
                x1,
            )

            label_y = y1 - 5

            if label_y < (text_height + 5):
                label_y = y1 + text_height + 5

            if label_x + text_width + 8 >= result.shape[1]:
                label_x = max(
                    2,
                    result.shape[1] - text_width - 10,
                )

            bg_top = max(
                0,
                label_y - text_height - 5,
            )

            bg_bottom = min(
                result.shape[0] - 1,
                label_y,
            )

            cv2.rectangle(
                result,
                (
                    label_x,
                    bg_top,
                ),
                (
                    label_x + text_width + 6,
                    bg_bottom,
                ),
                color,
                -1,
            )

            cv2.putText(
                result,
                label,
                (
                    label_x + 3,
                    label_y - 3,
                ),
                font,
                font_scale,
                (
                    255,
                    255,
                    255,
                ),
                thickness,
                cv2.LINE_AA,
            )

        return result

    # ========================================================
    # JPEG ENCODING
    # ========================================================

    @staticmethod
    def _encode_jpeg(
        image,
    ):
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success:
            raise RuntimeError("Failed to encode lesion overlay as JPEG.")

        return encoded.tobytes()

    # ========================================================
    # CLOUDINARY UPLOAD
    # ========================================================

    @staticmethod
    def _upload_overlay(
        image_bytes,
        screening_id,
    ):
        """
        Preserve the existing Cloudinary path.
        """

        result = cloudinary.uploader.upload(
            image_bytes,
            resource_type="image",
            public_id=("drishti-ai/" "screenings/" f"{screening_id}/" "lesion_overlay"),
            overwrite=True,
        )

        return {
            "url": result.get("secure_url"),
            "public_id": result.get("public_id"),
            "format": result.get("format"),
        }

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        image,
        screening_id=None,
    ):
        """
        Run ClementP lesion segmentation and create the same
        kind of result expected by ScreeningService.
        """

        image = self._validate_bgr_image(image)

        if not screening_id:
            raise ValueError("screening_id is required " "for lesion overlay upload.")

        original_height, original_width = image.shape[:2]

        # ----------------------------------------------------
        # ClementP segmentation
        # ----------------------------------------------------

        probabilities = self._segment(image)

        # ----------------------------------------------------
        # Retinal mask
        # ----------------------------------------------------

        retinal_mask = self._create_retinal_mask(image)

        # ----------------------------------------------------
        # Extract lesion regions
        # ----------------------------------------------------

        lesions: List[dict] = []

        for class_id in self.CLASS_NAMES:

            class_probability = probabilities[class_id]

            regions = self._get_regions(
                probability_map=(class_probability),
                class_id=class_id,
                retinal_mask=(retinal_mask),
            )

            lesions.extend(regions)

        # ----------------------------------------------------
        # Sort by confidence
        # ----------------------------------------------------

        lesions.sort(
            key=lambda item: (item["confidence"]),
            reverse=True,
        )

        # Prevent excessive weak regions from covering
        # the complete retina.
        MAX_LESIONS = 30

        lesions = lesions[:MAX_LESIONS]

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = self.create_overlay(
            image=image,
            lesions=lesions,
            alpha=0.28,
        )

        overlay_bytes = self._encode_jpeg(overlay)

        # Existing Cloudinary behavior is preserved.
        upload_result = self._upload_overlay(
            overlay_bytes,
            screening_id,
        )

        # ----------------------------------------------------
        # Public JSON
        # ----------------------------------------------------

        public_lesions = []

        for lesion in lesions:

            public_lesions.append(
                {
                    "type": lesion["type"],
                    "display_name": lesion["display_name"],
                    "class_id": lesion["class_id"],
                    "confidence": lesion["confidence"],
                    "peak_confidence": lesion["peak_confidence"],
                    "bbox": lesion["bbox"],
                    "area_pixels": lesion["area_pixels"],
                    "area_ratio": lesion["area_ratio"],
                    "center": lesion["center"],
                }
            )

        # ----------------------------------------------------
        # Per-class summary
        # ----------------------------------------------------

        class_summary = {}

        for class_id, class_name in self.CLASS_NAMES.items():

            class_lesions = [
                item for item in public_lesions if item["class_id"] == class_id
            ]

            class_summary[class_name] = {
                "display_name": self.CLASS_DISPLAY_NAMES[class_id],
                "count": len(class_lesions),
                "max_confidence": (
                    round(
                        max(
                            (item["confidence"] for item in class_lesions),
                            default=0.0,
                        ),
                        4,
                    )
                ),
            }

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return {
            "status": "success",
            "model": "ClementP fundus-lesions-toolkit",
            "model_architecture": "UNet + SEResNeXt50-32x4d",
            "model_training_variant": "ALL",
            "image_size": {
                "width": original_width,
                "height": original_height,
            },
            "confidence_threshold": self.confidence_threshold,
            "class_thresholds": self.CLASS_THRESHOLDS,
            "lesion_classes": [
                {
                    "class_id": class_id,
                    "name": self.CLASS_NAMES[class_id],
                    "display_name": self.CLASS_DISPLAY_NAMES[class_id],
                }
                for class_id in self.CLASS_NAMES
            ],
            "class_summary": class_summary,
            "lesion_count": len(public_lesions),
            "lesions": public_lesions,
            "overlay_url": upload_result["url"],
            "cloudinary": {
                "public_id": upload_result["public_id"],
                "format": upload_result["format"],
            },
        }


# ============================================================
# CREATE MODEL
# ============================================================


def create_lesion_model(
    num_classes=5,
    device=None,
):
    """
    Compatibility helper.

    Returns the ClementP segmentation model rather than
    the old custom U-Net.
    """

    if num_classes != 5:
        raise ValueError(
            "ClementP lesion model currently expects " "5 segmentation channels."
        )

    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = get_model(
        arch="unet",
        encoder="seresnext50_32x4d",
        train_datasets=Dataset.ALL,
        device=device,
        compile=False,
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# OPTIONAL LOCAL TEST
# ============================================================

if __name__ == "__main__":
    print("ClementP LesionDetector module loaded.")
