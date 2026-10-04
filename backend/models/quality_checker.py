import cv2
import numpy as np


class ImageQualityChecker:

    def __init__(
        self,
        min_width=224,
        min_height=224,
        blur_threshold=20.0,
        min_brightness=20.0,
        max_brightness=235.0,
        min_contrast=15.0
    ):

        self.min_width = min_width
        self.min_height = min_height

        self.blur_threshold = blur_threshold

        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        self.min_contrast = min_contrast

    # =========================================================
    # Resolution
    # =========================================================

    def check_resolution(self, image):

        height, width = image.shape[:2]

        passed = (
            width >= self.min_width
            and height >= self.min_height
        )

        return {
            "passed": bool(passed),
            "width": int(width),
            "height": int(height)
        }

    # =========================================================
    # Find approximate retinal field
    # =========================================================

    def get_retinal_region(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        # Threshold to remove very dark background
        mask = gray > 10

        coords = cv2.findNonZero(
            mask.astype(np.uint8)
        )

        if coords is None:

            return image

        x, y, w, h = cv2.boundingRect(
            coords
        )

        # Safety check
        if w < 100 or h < 100:

            return image

        cropped = image[
            y:y + h,
            x:x + w
        ]

        return cropped

    # =========================================================
    # Blur
    # =========================================================

    def check_blur(self, image):

        retinal_region = (
            self.get_retinal_region(image)
        )

        gray = cv2.cvtColor(
            retinal_region,
            cv2.COLOR_RGB2GRAY
        )

        # Slight resize for consistency
        gray = cv2.resize(
            gray,
            (512, 512)
        )

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        blur_score = float(
            laplacian.var()
        )

        passed = (
            blur_score >= self.blur_threshold
        )

        return {
            "passed": bool(passed),
            "blur_score": round(
                blur_score,
                2
            )
        }

    # =========================================================
    # Brightness
    # =========================================================

    def check_brightness(self, image):

        retinal_region = (
            self.get_retinal_region(image)
        )

        gray = cv2.cvtColor(
            retinal_region,
            cv2.COLOR_RGB2GRAY
        )

        brightness = float(
            np.mean(gray)
        )

        passed = (
            self.min_brightness
            <= brightness
            <= self.max_brightness
        )

        return {
            "passed": bool(passed),
            "brightness": round(
                brightness,
                2
            )
        }

    # =========================================================
    # Contrast
    # =========================================================

    def check_contrast(self, image):

        retinal_region = (
            self.get_retinal_region(image)
        )

        gray = cv2.cvtColor(
            retinal_region,
            cv2.COLOR_RGB2GRAY
        )

        contrast = float(
            np.std(gray)
        )

        passed = (
            contrast >= self.min_contrast
        )

        return {
            "passed": bool(passed),
            "contrast": round(
                contrast,
                2
            )
        }

    # =========================================================
    # Retina visibility
    # =========================================================

    def check_retina_visibility(self, image):

        height, width = image.shape[:2]

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        # Ignore extremely dark pixels
        mask = gray > 10

        visible_ratio = float(
            np.mean(mask)
        )

        # We only use this as a weak quality indicator.
        # A retinal image can contain a significant
        # amount of dark background.

        passed = (
            visible_ratio >= 0.10
        )

        return {
            "passed": bool(passed),
            "visibility_ratio": round(
                visible_ratio,
                4
            )
        }

    # =========================================================
    # Overall quality
    # =========================================================

    def analyze(self, image):

        resolution = (
            self.check_resolution(image)
        )

        blur = (
            self.check_blur(image)
        )

        brightness = (
            self.check_brightness(image)
        )

        contrast = (
            self.check_contrast(image)
        )

        retina_visibility = (
            self.check_retina_visibility(
                image
            )
        )

        checks = {
            "resolution": resolution,
            "blur": blur,
            "brightness": brightness,
            "contrast": contrast,
            "retina_visibility":
                retina_visibility
        }

        # =====================================================
        # Calculate weighted score
        # =====================================================

        scores = {
            "resolution":
                1.0
                if resolution["passed"]
                else 0.0,

            "blur":
                1.0
                if blur["passed"]
                else 0.0,

            "brightness":
                1.0
                if brightness["passed"]
                else 0.0,

            "contrast":
                1.0
                if contrast["passed"]
                else 0.0,

            "retina_visibility":
                1.0
                if retina_visibility["passed"]
                else 0.0
        }

        weights = {
            "resolution": 0.10,
            "blur": 0.35,
            "brightness": 0.20,
            "contrast": 0.25,
            "retina_visibility": 0.10
        }

        quality_score = sum(
            scores[key] * weights[key]
            for key in scores
        )

        quality_score = float(
            quality_score
        )

        # =====================================================
        # Status
        # =====================================================

        # Resolution failure is critical
        if not resolution["passed"]:

            status = "poor"

        # Extremely dark/bright image
        elif (
            brightness["brightness"] < 10
            or brightness["brightness"] > 245
        ):

            status = "poor"

        # Very blurry image
        elif (
            blur["blur_score"] < 10
        ):

            status = "poor"

        elif quality_score >= 0.70:

            status = "good"

        elif quality_score >= 0.45:

            status = "acceptable"

        else:

            status = "poor"

        return {
            "status": status,
            "score": round(
                quality_score,
                4
            ),
            "checks": checks
        }