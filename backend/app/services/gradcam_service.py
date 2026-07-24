import os
import uuid

import cv2
import numpy as np
import torch

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.services.model_service import model_service


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

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

    # Image used for visualization
    rgb_image = np.asarray(
        original_image
    ).astype(np.float32)

    rgb_image = rgb_image / 255.0

    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    input_tensor = model_service.prepare_image(
        original_image
    )

    # Grad-CAM needs gradients.
    # Do NOT use torch.inference_mode() here.
    input_tensor.requires_grad_(True)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_service.model.eval()

    # EfficientNet-B0 final feature block
    target_layers = [
        model_service.model.features[-1]
    ]

    # Tell Grad-CAM which class should be explained
    targets = [
        ClassifierOutputTarget(
            int(predicted_class_index)
        )
    ]

    # --------------------------------------------------------
    # Generate Grad-CAM
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

        if grayscale_cam is None:
            raise RuntimeError(
                "Grad-CAM returned no heatmap."
            )

        grayscale_cam = grayscale_cam[0]

        # ----------------------------------------------------
        # Overlay heatmap
        # ----------------------------------------------------

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        filename = (
            f"gradcam_{uuid.uuid4().hex}.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        visualization_bgr = cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )

        success = cv2.imwrite(
            output_path,
            visualization_bgr
        )

        if not success:
            raise RuntimeError(
                "Failed to save Grad-CAM image."
            )

        if not os.path.exists(output_path):
            raise RuntimeError(
                "Grad-CAM output file was not created."
            )

        return {
            "gradcam_path": output_path,
            "gradcam_filename": filename
        }

    finally:

        # Release hooks/resources when supported
        if hasattr(cam, "activations_and_grads"):
            try:
                cam.activations_and_grads.release()
            except Exception:
                pass
