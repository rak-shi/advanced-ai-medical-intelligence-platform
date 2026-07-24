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

from app.services.model_service import (
    model_service,
    DEVICE
)


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


def generate_gradcam(image_path):

    # --------------------------------------------
    # Load original image
    # --------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    original_image = original_image.resize(
        (
            model_service.image_size,
            model_service.image_size
        )
    )

    rgb_image = np.array(
        original_image
    ).astype(np.float32) / 255.0


    # --------------------------------------------
    # Prepare model input
    # --------------------------------------------

    input_tensor = (
        model_service.prepare_image(
            original_image
        )
    )


    # --------------------------------------------
    # Get prediction
    # --------------------------------------------

    with torch.no_grad():

        output = model_service.model(
            input_tensor
        )

        predicted_class = (
            output.argmax(dim=1).item()
        )


    # --------------------------------------------
    # EfficientNet target layer
    # --------------------------------------------

    target_layers = [
        model_service.model.features[-1]
    ]


    # --------------------------------------------
    # Grad-CAM
    # --------------------------------------------

    cam = GradCAM(
        model=model_service.model,
        target_layers=target_layers
    )


    grayscale_cam = cam(
        input_tensor=input_tensor
    )


    grayscale_cam = grayscale_cam[0]


    # --------------------------------------------
    # Overlay heatmap
    # --------------------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # --------------------------------------------
    # Save result
    # --------------------------------------------

    filename = (
        f"gradcam_{uuid.uuid4().hex}.jpg"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )


    return {
        "gradcam_path": output_path,

        "predicted_class":
            model_service.class_names[
                predicted_class
            ]
    }
