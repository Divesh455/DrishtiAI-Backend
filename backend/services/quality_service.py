from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from backend.models.quality_checker import (
    ImageQualityChecker
)


class QualityService:
    """
    Combined image-quality service for DrishtiAI.

    Two quality layers are used:

    1. Technical quality checker
       - Resolution
       - Blur
       - Brightness
       - Contrast
       - Retinal visibility

    2. Fundus Suitability AI
       - Fundus
       - Not fundus

    The image is allowed to continue to DR screening only
    when it is technically acceptable AND identified as a
    fundus image.
    """

    def __init__(self):

        # ----------------------------------------------------
        # Existing technical quality checker
        # ----------------------------------------------------

        self.checker = ImageQualityChecker()

        # ----------------------------------------------------
        # Fundus Suitability model configuration
        # ----------------------------------------------------

        project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
            .parent
        )

        self.model_path = (
            project_root
            / "weights"
            / "quality_model.pth"
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.fundus_threshold = 0.50

        # ----------------------------------------------------
        # Image preprocessing
        # ----------------------------------------------------

        self.fundus_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

        # ----------------------------------------------------
        # Load trained Fundus Suitability AI
        # ----------------------------------------------------

        self.fundus_model = (
            self._load_fundus_model()
        )

    # ========================================================
    # LOAD FUNDUS SUITABILITY MODEL
    # ========================================================

    def _load_fundus_model(self):
        """
        Load the trained EfficientNet-B0 Fundus Suitability
        classifier from weights/quality_model.pth.
        """

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Fundus Suitability model not found: "
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # Build the same architecture used during training.
        #
        # weights=None is intentional here because the trained
        # checkpoint already contains the model parameters.
        # ----------------------------------------------------

        model = models.efficientnet_b0(
            weights=None
        )

        in_features = (
            model.classifier[1].in_features
        )

        model.classifier = nn.Sequential(
            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                in_features,
                1,
            ),
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        if (
            "model_state_dict"
            not in checkpoint
        ):

            raise RuntimeError(
                "Invalid Fundus Suitability checkpoint. "
                "Expected 'model_state_dict'."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model = model.to(
            self.device
        )

        model.eval()

        return model

    # ========================================================
    # FUNDUS SUITABILITY PREDICTION
    # ========================================================

    def _predict_fundus_suitability(
        self,
        image: Image.Image,
    ):
        """
        Predict whether the image is a fundus image.

        Returns:

        {
            "label": "fundus" / "not_fundus",
            "is_fundus": bool,
            "confidence": float,
            "fundus_probability": float,
            "not_fundus_probability": float,
            "threshold": float
        }
        """

        image = image.convert(
            "RGB"
        )

        tensor = self.fundus_transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            logits = self.fundus_model(
                tensor
            ).squeeze(1)

            fundus_probability = (
                torch.sigmoid(
                    logits
                )
                .item()
            )

        not_fundus_probability = (
            1.0
            - fundus_probability
        )

        is_fundus = (
            fundus_probability
            >= self.fundus_threshold
        )

        if is_fundus:

            label = "fundus"

            confidence = (
                fundus_probability
            )

        else:

            label = "not_fundus"

            confidence = (
                not_fundus_probability
            )

        return {
            "label": label,

            "is_fundus": bool(
                is_fundus
            ),

            "confidence": float(
                confidence
            ),

            "fundus_probability": float(
                fundus_probability
            ),

            "not_fundus_probability": float(
                not_fundus_probability
            ),

            "threshold": float(
                self.fundus_threshold
            ),
        }

    # ========================================================
    # MAIN QUALITY CHECK
    # ========================================================

    def check_pil_image(
        self,
        image: Image.Image
    ):
        """
        Run both technical quality analysis and
        Fundus Suitability AI.
        """

        # ----------------------------------------------------
        # Standardize image
        # ----------------------------------------------------

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # Existing technical quality analysis
        # ----------------------------------------------------

        image_array = np.array(
            image
        )

        technical_result = (
            self.checker.analyze(
                image_array
            )
        )

        # ----------------------------------------------------
        # Learned Fundus Suitability prediction
        # ----------------------------------------------------

        fundus_result = (
            self._predict_fundus_suitability(
                image
            )
        )

        # ----------------------------------------------------
        # Combine both quality systems
        # ----------------------------------------------------

        result = dict(
            technical_result
        )

        result[
            "fundus_suitability"
        ] = fundus_result

        # ----------------------------------------------------
        # Final suitability decision
        # ----------------------------------------------------

        if not fundus_result[
            "is_fundus"
        ]:

            result[
                "screening_ready"
            ] = False

            result[
                "status"
            ] = "poor"

            result[
                "quality_decision"
            ] = "reject"

            result[
                "decision_reason"
            ] = (
                "The uploaded image was not "
                "identified as a suitable fundus "
                "image."
            )

        else:

            result[
                "screening_ready"
            ] = (
                technical_result[
                    "status"
                ]
                in [
                    "good",
                    "acceptable",
                ]
            )

            result[
                "quality_decision"
            ] = (
                "accept"
                if result[
                    "screening_ready"
                ]
                else "reject"
            )

            if result[
                "screening_ready"
            ]:

                result[
                    "decision_reason"
                ] = (
                    "The image is identified as "
                    "a fundus image and its "
                    "technical quality is "
                    "sufficient for screening."
                )

            else:

                result[
                    "decision_reason"
                ] = (
                    "The image is identified as "
                    "a fundus image, but its "
                    "technical quality is "
                    "insufficient for screening."
                )

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        result[
            "recommendation"
        ] = self._generate_recommendation(
            result
        )

        return result

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    def _generate_recommendation(
        self,
        result
    ):
        """
        Generate a user-facing recommendation.

        Fundus suitability is checked first because a normal
        photograph, selfie, document, etc. should not be
        described merely as a blurry retinal image.
        """

        # ----------------------------------------------------
        # Not a fundus image
        # ----------------------------------------------------

        fundus_result = result.get(
            "fundus_suitability",
            {}
        )

        if not fundus_result.get(
            "is_fundus",
            False
        ):

            return (
                "The uploaded image does not "
                "appear to be a fundus photograph. "
                "Please capture and upload a clear "
                "retinal image."
            )

        # ----------------------------------------------------
        # Technical quality checks
        # ----------------------------------------------------

        if result["status"] == "good":

            return (
                "Fundus image detected and image "
                "quality is sufficient for screening."
            )

        checks = result[
            "checks"
        ]

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------

        if not checks[
            "resolution"
        ]["passed"]:

            return (
                "Fundus image detected, but the "
                "image resolution is too low. "
                "Please capture a higher-resolution "
                "retinal image."
            )

        # ----------------------------------------------------
        # Blur
        # ----------------------------------------------------

        if not checks[
            "blur"
        ]["passed"]:

            return (
                "Fundus image detected, but the "
                "image is too blurry. Hold the "
                "camera steady and capture another "
                "retinal image."
            )

        # ----------------------------------------------------
        # Brightness
        # ----------------------------------------------------

        if not checks[
            "brightness"
        ]["passed"]:

            brightness = checks[
                "brightness"
            ]["brightness"]

            if brightness < 35:

                return (
                    "Fundus image detected, but "
                    "the image is too dark. Improve "
                    "illumination and capture another "
                    "retinal image."
                )

            return (
                "Fundus image detected, but the "
                "image is too bright. Reduce "
                "excessive illumination and capture "
                "another image."
            )

        # ----------------------------------------------------
        # Contrast
        # ----------------------------------------------------

        if not checks[
            "contrast"
        ]["passed"]:

            return (
                "Fundus image detected, but image "
                "contrast is too low. Please capture "
                "a clearer retinal image."
            )

        # ----------------------------------------------------
        # Retinal visibility
        # ----------------------------------------------------

        if not checks[
            "retina_visibility"
        ]["passed"]:

            return (
                "Fundus image detected, but the "
                "retinal region is not clearly visible. "
                "Please reposition the camera and "
                "capture the retina again."
            )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        return (
            "Fundus image detected, but image "
            "quality is insufficient. Please capture "
            "another retinal image."
        )