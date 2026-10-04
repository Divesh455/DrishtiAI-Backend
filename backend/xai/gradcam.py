# backend/xai/gradcam.py

from __future__ import annotations

import io
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

import cloudinary
import cloudinary.uploader

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)


class GradCAMExplainer:
    """
    Grad-CAM++ explainability module for DrishtiAI.

    Purpose:
        Show regions that contributed to the DR classifier
        prediction.

    IMPORTANT:
        Grad-CAM++ is classifier attention.
        It is NOT a lesion detector.

    Output:
        - original image
        - heatmap
        - transparent Grad-CAM overlay
        - CAM array
        - target class
        - predicted class

    Cloudinary:
        Images are uploaded directly from memory.
        No local image files are created.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        image_size: int = 224,
    ):
        self.model = model

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.image_size = int(
            image_size
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        self.target_layer = (
            self._find_target_layer()
        )

        print(
            f"[GradCAM] Device: {self.device}"
        )

        print(
            f"[GradCAM] Image size: "
            f"{self.image_size}"
        )

        print(
            "[GradCAM] Target layer:"
        )

        print(
            self.target_layer
        )

    # ============================================================
    # FIND TARGET LAYER
    # ============================================================

    def _find_target_layer(self):
        """
        Find the Grad-CAM target layer.

        For the EfficientNet-based DR classifier, features[5]
        provides a useful balance between spatial detail and
        semantic information.

        Falls back to the last convolutional layer if needed.
        """

        backbone = self.model

        # Your DR model may wrap EfficientNet inside
        # self.model.model.
        if hasattr(
            backbone,
            "model",
        ):

            backbone = backbone.model

        if not hasattr(
            backbone,
            "features",
        ):

            raise AttributeError(
                "Could not find EfficientNet "
                "'features' in DR model."
            )

        features = backbone.features

        # Preferred target layer.
        if len(features) > 5:

            return features[5]

        # Fallback.
        for index in range(
            len(features) - 1,
            -1,
            -1,
        ):

            module = features[index]

            last_conv = None

            for child in module.modules():

                if isinstance(
                    child,
                    (
                        torch.nn.Conv1d,
                        torch.nn.Conv2d,
                        torch.nn.Conv3d,
                    ),
                ):

                    last_conv = child

            if last_conv is not None:

                return last_conv

        raise RuntimeError(
            "Could not find a convolutional "
            "layer for Grad-CAM."
        )

    # ============================================================
    # IMAGE → RGB UINT8
    # ============================================================

    @staticmethod
    def _to_rgb_uint8(
        image: Any,
    ) -> np.ndarray:
        """
        Convert PIL / NumPy / Tensor image to RGB uint8.
        """

        # --------------------------------------------------------
        # PIL
        # --------------------------------------------------------

        if isinstance(
            image,
            Image.Image,
        ):

            return np.array(
                image.convert("RGB")
            )

        # --------------------------------------------------------
        # Tensor
        # --------------------------------------------------------

        if isinstance(
            image,
            torch.Tensor,
        ):

            tensor = (
                image.detach()
                .cpu()
            )

            if tensor.ndim == 4:

                tensor = tensor[0]

            if (
                tensor.ndim == 3
                and tensor.shape[0] in (1, 3)
            ):

                tensor = tensor.permute(
                    1,
                    2,
                    0,
                )

            array = tensor.numpy()

            if array.max() <= 1.0:

                array = (
                    array * 255.0
                )

            array = np.clip(
                array,
                0,
                255,
            ).astype(
                np.uint8
            )

            if (
                array.ndim == 3
                and array.shape[2] == 1
            ):

                array = cv2.cvtColor(
                    array,
                    cv2.COLOR_GRAY2RGB,
                )

            return array

        # --------------------------------------------------------
        # NumPy
        # --------------------------------------------------------

        array = np.asarray(
            image
        )

        if array.ndim == 2:

            array = cv2.cvtColor(
                array,
                cv2.COLOR_GRAY2RGB,
            )

        if array.dtype != np.uint8:

            if array.max() <= 1.0:

                array = (
                    array * 255.0
                )

            array = np.clip(
                array,
                0,
                255,
            ).astype(
                np.uint8
            )

        return array

    # ============================================================
    # RETINAL MASK
    # ============================================================

    @staticmethod
    def _create_retinal_mask(
        image_rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the visible fundus region.

        This prevents Grad-CAM from highlighting black
        background outside the retina.
        """

        height, width = (
            image_rgb.shape[:2]
        )

        # --------------------------------------------------------
        # HSV
        # --------------------------------------------------------

        hsv = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2HSV,
        )

        saturation = hsv[
            :,
            :,
            1,
        ]

        value = hsv[
            :,
            :,
            2,
        ]

        # --------------------------------------------------------
        # Candidate retinal pixels
        # --------------------------------------------------------

        candidate = (
            (
                saturation > 15
            )
            &
            (
                value > 10
            )
        ).astype(
            np.uint8
        ) * 255

        # --------------------------------------------------------
        # Morphological cleanup
        # --------------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                15,
                15,
            ),
        )

        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            kernel,
        )

        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            kernel,
        )

        # --------------------------------------------------------
        # Largest connected region
        # --------------------------------------------------------

        (
            num_labels,
            labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        if num_labels > 1:

            areas = stats[
                1:,
                cv2.CC_STAT_AREA,
            ]

            largest_index = (
                1
                + int(
                    np.argmax(
                        areas
                    )
                )
            )

            largest_area = stats[
                largest_index,
                cv2.CC_STAT_AREA,
            ]

            image_area = (
                height * width
            )

            # Require a meaningful fundus region.
            if (
                largest_area
                > image_area * 0.15
            ):

                mask[
                    labels
                    == largest_index
                ] = 255

        # --------------------------------------------------------
        # Fallback
        # --------------------------------------------------------

        if mask.max() == 0:

            center_x = (
                width // 2
            )

            center_y = (
                height // 2
            )

            radius = int(
                min(
                    width,
                    height,
                )
                * 0.47
            )

            cv2.circle(
                mask,
                (
                    center_x,
                    center_y,
                ),
                radius,
                255,
                -1,
            )

        # --------------------------------------------------------
        # Smooth boundary
        # --------------------------------------------------------

        mask = cv2.GaussianBlur(
            mask,
            (
                21,
                21,
            ),
            0,
        )

        mask = (
            mask.astype(
                np.float32
            )
            / 255.0
        )

        return np.clip(
            mask,
            0.0,
            1.0,
        )

    # ============================================================
    # PREPROCESS
    # ============================================================

    def _preprocess(
        self,
        image_rgb: np.ndarray,
    ) -> torch.Tensor:
        """
        Resize and ImageNet-normalize image.
        """

        resized = cv2.resize(
            image_rgb,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        array = (
            resized.astype(
                np.float32
            )
            / 255.0
        )

        # ImageNet normalization.
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

        array = (
            array - mean
        ) / std

        array = np.transpose(
            array,
            (
                2,
                0,
                1,
            ),
        )

        tensor = torch.tensor(
            array,
            dtype=torch.float32,
        )

        tensor = tensor.unsqueeze(
            0
        )

        return tensor.to(
            self.device
        )

    # ============================================================
    # MODEL LOGITS
    # ============================================================

    def _get_logits(
        self,
        output,
    ):
        """
        Handle common model output formats.
        """

        if isinstance(
            output,
            tuple,
        ):

            return output[0]

        if isinstance(
            output,
            dict,
        ):

            if "logits" not in output:

                raise RuntimeError(
                    "Model returned a dictionary "
                    "without 'logits'."
                )

            return output[
                "logits"
            ]

        return output

    # ============================================================
    # ADAPTIVE CAM NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_cam(
        cam: np.ndarray,
        retinal_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize Grad-CAM using retinal-region
        percentile normalization.

        This removes weak diffuse activation while
        preserving meaningful high activation.
        """

        cam = np.asarray(
            cam,
            dtype=np.float32,
        )

        cam = np.nan_to_num(
            cam,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Grad-CAM is positive activation only.
        cam = np.maximum(
            cam,
            0.0,
        )

        valid = cam[
            retinal_mask > 0.35
        ]

        if valid.size < 20:

            valid = cam.flatten()

        if valid.size == 0:

            return np.zeros_like(
                cam
            )

        # --------------------------------------------------------
        # Percentile scaling
        # --------------------------------------------------------

        low = np.percentile(
            valid,
            30,
        )

        high = np.percentile(
            valid,
            98,
        )

        if high <= low + 1e-8:

            return np.zeros_like(
                cam
            )

        cam = (
            cam - low
        ) / (
            high - low
        )

        cam = np.clip(
            cam,
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # Smooth
        # --------------------------------------------------------

        cam = cv2.GaussianBlur(
            cam,
            (
                5,
                5,
            ),
            0,
        )

        # --------------------------------------------------------
        # Retinal mask
        # --------------------------------------------------------

        cam *= retinal_mask

        # --------------------------------------------------------
        # Gentle nonlinear enhancement
        #
        # Do NOT create hard segmentation-like regions.
        # --------------------------------------------------------

        cam = np.power(
            cam,
            0.85,
        )

        # --------------------------------------------------------
        # Final normalization
        # --------------------------------------------------------

        maximum = cam.max()

        if maximum > 1e-8:

            cam /= maximum

        return np.clip(
            cam,
            0.0,
            1.0,
        )

    # ============================================================
    # HEATMAP
    # ============================================================

    @staticmethod
    def _create_heatmap(
        cam: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Create RGB heatmap.
        """

        cam_resized = cv2.resize(
            cam,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        cam_resized = np.clip(
            cam_resized,
            0.0,
            1.0,
        )

        cam_uint8 = (
            cam_resized
            * 255.0
        ).astype(
            np.uint8
        )

        heatmap_bgr = (
            cv2.applyColorMap(
                cam_uint8,
                cv2.COLORMAP_JET,
            )
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return heatmap_rgb

    # ============================================================
    # OVERLAY
    # ============================================================

    @staticmethod
    def _create_overlay(
        original_rgb: np.ndarray,
        heatmap_rgb: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.40,
    ) -> np.ndarray:
        """
        Create a smooth transparent Grad-CAM overlay.

        Strong activation:
            stronger heatmap

        Weak activation:
            mostly original retina
        """

        original = (
            original_rgb.astype(
                np.float32
            )
            / 255.0
        )

        heatmap = (
            heatmap_rgb.astype(
                np.float32
            )
            / 255.0
        )

        cam_resized = cv2.resize(
            cam,
            (
                original.shape[1],
                original.shape[0],
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        cam_resized = np.clip(
            cam_resized,
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # Smooth attention strength
        # --------------------------------------------------------

        strength = np.power(
            cam_resized,
            0.80,
        )

        # --------------------------------------------------------
        # Adaptive transparency
        # --------------------------------------------------------

        weight = (
            strength[..., None]
            * alpha
        )

        overlay = (
            original
            * (
                1.0
                - weight
            )
            +
            heatmap
            * weight
        )

        overlay = np.clip(
            overlay,
            0.0,
            1.0,
        )

        return (
            overlay
            * 255.0
        ).astype(
            np.uint8
        )

    # ============================================================
    # GENERATE
    # ============================================================

    def generate(
        self,
        image: Any,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM++.

        Returns:

            {
                "heatmap": RGB image,
                "overlay": RGB image,
                "original": RGB image,
                "cam": float array,
                "target_class": int,
                "predicted_class": int
            }
        """

        # --------------------------------------------------------
        # Convert input
        # --------------------------------------------------------

        original_rgb = (
            self._to_rgb_uint8(
                image
            )
        )

        height, width = (
            original_rgb.shape[:2]
        )

        # --------------------------------------------------------
        # Retinal mask
        # --------------------------------------------------------

        retinal_mask = (
            self._create_retinal_mask(
                original_rgb
            )
        )

        # --------------------------------------------------------
        # Model input
        # --------------------------------------------------------

        input_tensor = (
            self._preprocess(
                original_rgb
            )
        )

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        with torch.no_grad():

            output = self.model(
                input_tensor
            )

            logits = self._get_logits(
                output
            )

        if logits.ndim == 1:

            logits = logits.unsqueeze(
                0
            )

        predicted_class = int(
            torch.argmax(
                logits,
                dim=1,
            ).item()
        )

        if target_class is None:

            target_class = (
                predicted_class
            )

        target_class = int(
            target_class
        )

        # --------------------------------------------------------
        # Grad-CAM++
        # --------------------------------------------------------

        targets = [
            ClassifierOutputTarget(
                target_class
            )
        ]

        cam_algorithm = (
            GradCAMPlusPlus(
                model=self.model,
                target_layers=[
                    self.target_layer
                ],
            )
        )

        try:

            grayscale_cam = (
                cam_algorithm(
                    input_tensor=input_tensor,
                    targets=targets,
                )[0]
            )

        finally:

            # Release hooks.
            try:

                cam_algorithm.activations_and_grads.release()

            except Exception:

                pass

        # --------------------------------------------------------
        # Normalize CAM
        # --------------------------------------------------------

        normalized_cam = (
            self._normalize_cam(
                grayscale_cam,
                retinal_mask=(
                    cv2.resize(
                        retinal_mask,
                        (
                            grayscale_cam.shape[1],
                            grayscale_cam.shape[0],
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                ),
            )
        )

        # --------------------------------------------------------
        # Resize CAM to original image
        # --------------------------------------------------------

        normalized_cam_full = (
            cv2.resize(
                normalized_cam,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        )

        # Apply full-resolution retinal mask.
        normalized_cam_full *= (
            retinal_mask
        )

        normalized_cam_full = np.clip(
            normalized_cam_full,
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # Heatmap
        # --------------------------------------------------------

        heatmap = (
            self._create_heatmap(
                normalized_cam_full,
                height,
                width,
            )
        )

        # --------------------------------------------------------
        # Overlay
        # --------------------------------------------------------

        overlay = (
            self._create_overlay(
                original_rgb,
                heatmap,
                normalized_cam_full,
                alpha=0.40,
            )
        )

        # --------------------------------------------------------
        # Outside retina = original image
        # --------------------------------------------------------

        outside = (
            retinal_mask < 0.10
        )

        heatmap[outside] = (
            original_rgb[outside]
        )

        overlay[outside] = (
            original_rgb[outside]
        )

        return {

            "heatmap":
                heatmap,

            "overlay":
                overlay,

            "original":
                original_rgb,

            "cam":
                normalized_cam_full,

            "target_class":
                target_class,

            "predicted_class":
                predicted_class,
        }

    # ============================================================
    # JPEG ENCODER
    # ============================================================

    @staticmethod
    def _encode_jpeg(
        image_rgb: np.ndarray,
        quality: int = 95,
    ) -> bytes:
        """
        Convert RGB image to JPEG bytes.
        """

        image_rgb = np.asarray(
            image_rgb,
            dtype=np.uint8,
        )

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        success, encoded = (
            cv2.imencode(
                ".jpg",
                image_bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    quality,
                ],
            )
        )

        if not success:

            raise RuntimeError(
                "Failed to encode "
                "Grad-CAM image."
            )

        return encoded.tobytes()

    # ============================================================
    # CLOUDINARY UPLOAD
    # ============================================================

    @staticmethod
    def _upload_bytes(
        image_bytes: bytes,
        public_id: str,
    ) -> Dict[str, Any]:
        """
        Upload image bytes directly to Cloudinary.

        No local file is created.
        """

        result = (
            cloudinary.uploader.upload(
                io.BytesIO(
                    image_bytes
                ),
                public_id=public_id,
                resource_type="image",
                overwrite=True,
                format="jpg",
            )
        )

        return result

    # ============================================================
    # UPLOAD RESULTS
    # ============================================================

    def upload_results(
        self,
        result: Dict[str, Any],
        screening_id: str,
    ) -> Dict[str, Any]:
        """
        Upload:

            original
            heatmap
            Grad-CAM overlay

        to Cloudinary.
        """

        if not screening_id:

            raise ValueError(
                "screening_id is required "
                "for Cloudinary upload."
            )

        base_path = (
            "drishti-ai/"
            "screenings/"
            f"{screening_id}"
        )

        # --------------------------------------------------------
        # ORIGINAL
        # --------------------------------------------------------

        original_bytes = (
            self._encode_jpeg(
                result["original"]
            )
        )

        original_result = (
            self._upload_bytes(
                original_bytes,
                f"{base_path}/original",
            )
        )

        # --------------------------------------------------------
        # HEATMAP
        # --------------------------------------------------------

        heatmap_bytes = (
            self._encode_jpeg(
                result["heatmap"]
            )
        )

        heatmap_result = (
            self._upload_bytes(
                heatmap_bytes,
                f"{base_path}/heatmap",
            )
        )

        # --------------------------------------------------------
        # OVERLAY
        # --------------------------------------------------------

        overlay_bytes = (
            self._encode_jpeg(
                result["overlay"]
            )
        )

        overlay_result = (
            self._upload_bytes(
                overlay_bytes,
                f"{base_path}/gradcam_overlay",
            )
        )

        return {

            "original_url":
                original_result.get(
                    "secure_url"
                ),

            "original_public_id":
                original_result.get(
                    "public_id"
                ),

            "heatmap_url":
                heatmap_result.get(
                    "secure_url"
                ),

            "heatmap_public_id":
                heatmap_result.get(
                    "public_id"
                ),

            "overlay_url":
                overlay_result.get(
                    "secure_url"
                ),

            "overlay_public_id":
                overlay_result.get(
                    "public_id"
                ),
        }

    # ============================================================
    # GENERATE + UPLOAD
    # ============================================================

    def explain_and_upload(
        self,
        image: Any,
        screening_id: str,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM++ and upload all images.
        """

        result = self.generate(
            image=image,
            target_class=target_class,
        )

        uploaded = (
            self.upload_results(
                result=result,
                screening_id=screening_id,
            )
        )

        return {
            **result,
            **uploaded,
        }