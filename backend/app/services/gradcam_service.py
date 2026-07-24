import gc
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
    image_path: str,
    predicted_class_index: int | None = None
):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    cam = None
    input_tensor = None

    try:

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        with Image.open(image_path) as image:

            original_image = image.convert("RGB")

            original_image = original_image.resize(
                (
                    model_service.image_size,
                    model_service.image_size
                )
            )

        # Image used for heatmap overlay
        rgb_image = np.asarray(
            original_image
        ).astype(np.float32)

        rgb_image /= 255.0


        # ====================================================
        # PREPARE MODEL INPUT
        # ====================================================

        input_tensor = model_service.prepare_image(
            original_image
        )

        # Grad-CAM needs gradients.
        input_tensor.requires_grad_(True)


        # ====================================================
        # DETERMINE TARGET CLASS
        # ====================================================

        if predicted_class_index is None:

            # Only needed when caller didn't provide class.
            with torch.no_grad():

                output = model_service.model(
                    input_tensor
                )

                predicted_class_index = int(
                    torch.argmax(
                        output,
                        dim=1
                    ).item()
                )

            del output


        # ====================================================
        # TARGET LAYER
        # EfficientNet-B0 final feature block
        # ====================================================

        target_layers = [
            model_service.model.features[-1]
        ]


        # ====================================================
        # TARGET
        # ====================================================

        targets = [
            ClassifierOutputTarget(
                predicted_class_index
            )
        ]


        # ====================================================
        # CREATE GRAD-CAM
        # ====================================================

        cam = GradCAM(
            model=model_service.model,
            target_layers=target_layers
        )


        # ====================================================
        # GENERATE CAM
        # ====================================================

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )

        grayscale_cam = grayscale_cam[0]


        # ====================================================
        # CREATE VISUALIZATION
        # ====================================================

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )


        # ====================================================
        # SAVE OUTPUT
        # ====================================================

        filename = (
            f"gradcam_{uuid.uuid4().hex}.jpg"
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
                "OpenCV failed to save Grad-CAM image."
            )


        # ====================================================
        # RETURN
        # ====================================================

        return {
            "gradcam_path": output_path,
            "gradcam_filename": filename,
            "predicted_class_index":
                predicted_class_index,
            "predicted_class":
                str(
                    model_service.class_names[
                        predicted_class_index
                    ]
                )
        }


    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        if cam is not None:

            try:
                if hasattr(cam, "__exit__"):
                    cam.__exit__(
                        None,
                        None,
                        None
                    )
            except Exception:
                pass

        del cam
        del input_tensor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
