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
    model_service
)


# ============================================================
# DIRECTORIES
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
    image_path
):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    with Image.open(
        image_path
    ) as image:

        original_image = image.convert(
            "RGB"
        )


    original_image = (
        original_image.resize(
            (
                model_service.image_size,
                model_service.image_size
            )
        )
    )


    rgb_image = np.asarray(
        original_image
    ).astype(
        np.float32
    )


    rgb_image = (
        rgb_image / 255.0
    )


    # ========================================================
    # MODEL INPUT
    # ========================================================

    input_tensor = (
        model_service.prepare_image(
            original_image
        )
    )


    # ========================================================
    # PREDICT CLASS
    # ========================================================

    with torch.inference_mode():

        output = (
            model_service.model(
                input_tensor
            )
        )


        predicted_class_index = int(
            output.argmax(
                dim=1
            ).item()
        )


    # ========================================================
    # TARGET LAYER
    # ========================================================

    target_layers = [

        model_service.model.features[-1]

    ]


    # ========================================================
    # GRAD-CAM
    # ========================================================

    cam = GradCAM(
        model=model_service.model,
        target_layers=target_layers
    )


    try:

        grayscale_cam = cam(
            input_tensor=input_tensor
        )


        grayscale_cam = (
            grayscale_cam[0]
        )


        # ====================================================
        # VISUALIZATION
        # ====================================================

        visualization = (
            show_cam_on_image(
                rgb_image,
                grayscale_cam,
                use_rgb=True
            )
        )


        # ====================================================
        # SAVE
        # ====================================================

        filename = (
            "gradcam_"
            f"{uuid.uuid4().hex}"
            ".jpg"
        )


        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )


        saved = cv2.imwrite(
            output_path,
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR
            )
        )


        if not saved:

            raise RuntimeError(
                "OpenCV failed to save "
                "Grad-CAM image."
            )


        return {

            "gradcam_path":
                output_path,

            "predicted_class":
                str(
                    model_service.class_names[
                        predicted_class_index
                    ]
                )
        }


    finally:

        # Different pytorch-grad-cam versions
        # handle cleanup differently.
        if hasattr(
            cam,
            "__exit__"
        ):

            try:

                cam.__exit__(
                    None,
                    None,
                    None
                )

            except Exception:

                pass
