import os
import uuid

import cv2
import numpy as np
import torch

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)

from app.services.model_service import model_service


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "gradcam"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def generate_gradcam(
    image_path: str,
    predicted_class_index: int
):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    with Image.open(image_path) as image:
        original_image = image.convert("RGB")

    original_image = original_image.resize(
        (
            model_service.image_size,
            model_service.image_size
        )
    )

    # Image for Grad-CAM visualization
    rgb_image = np.asarray(
        original_image
    ).astype(np.float32)

    rgb_image = rgb_image / 255.0

    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    input_tensor = (
        model_service.prepare_image(
            original_image
        )
    )

    # --------------------------------------------------------
    # Target convolution layer
    # --------------------------------------------------------

    target_layers = [
        model_service.model.features[-1]
    ]

    # --------------------------------------------------------
    # Target class
    # --------------------------------------------------------

    targets = [
        ClassifierOutputTarget(
            predicted_class_index
        )
    ]

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    cam = GradCAM(
        model=model_service.model,
        target_layers=target_layers
    )

    try:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )

        grayscale_cam = grayscale_cam[0]

        # ----------------------------------------------------
        # Create heatmap overlay
        # ----------------------------------------------------

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        # ----------------------------------------------------
        # Save output
        # ----------------------------------------------------

        filename = (
            f"gradcam_{uuid.uuid4().hex}.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        success = cv2.imwrite(
            output_path,
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR
            )
        )

        if not success:
            raise RuntimeError(
                "Failed to save Grad-CAM image."
            )

        return {
            "gradcam_filename": filename,
            "gradcam_path": output_path
        }

    finally:

        if hasattr(cam, "__exit__"):
            try:
                cam.__exit__(
                    None,
                    None,
                    None
                )
            except Exception:
                pass
