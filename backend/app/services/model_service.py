import os

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "trained_model",
    "pneumonia_efficientnet_b0.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")


# ============================================================
# MODEL SERVICE
# ============================================================

class PneumoniaModelService:

    def __init__(self):

        # ----------------------------------------------------
        # Check model file
        # ----------------------------------------------------

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        print("Loading pneumonia model...")


        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )


        if not isinstance(checkpoint, dict):

            raise ValueError(
                "Model checkpoint must be a dictionary."
            )


        if "class_names" not in checkpoint:

            raise ValueError(
                "class_names missing from model checkpoint."
            )


        if "model_state_dict" not in checkpoint:

            raise ValueError(
                "model_state_dict missing from checkpoint."
            )


        self.class_names = checkpoint[
            "class_names"
        ]


        self.image_size = checkpoint.get(
            "image_size",
            224
        )


        print(
            f"Classes: {self.class_names}"
        )

        print(
            f"Image size: {self.image_size}"
        )


        # ----------------------------------------------------
        # Build EfficientNet-B0
        # ----------------------------------------------------

        self.model = models.efficientnet_b0(
            weights=None
        )


        number_features = (
            self.model.classifier[1].in_features
        )


        self.model.classifier[1] = nn.Linear(
            number_features,
            len(self.class_names)
        )


        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        self.model.to(DEVICE)

        self.model.eval()


        # ----------------------------------------------------
        # Image preprocessing
        # ----------------------------------------------------

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.image_size,
                        self.image_size
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225
                    ]
                )
            ]
        )


        print(
            "Medical imaging model loaded successfully."
        )


    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    def prepare_image(self, image):

        if isinstance(image, str):

            if not os.path.exists(image):

                raise FileNotFoundError(
                    f"Image not found: {image}"
                )

            with Image.open(image) as loaded_image:

                image = loaded_image.convert(
                    "RGB"
                )

        elif isinstance(image, Image.Image):

            image = image.convert(
                "RGB"
            )

        else:

            raise TypeError(
                "Image must be a file path "
                "or PIL Image."
            )


        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        return tensor.to(
            DEVICE
        )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, image):

        input_tensor = self.prepare_image(
            image
        )


        with torch.inference_mode():

            output = self.model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted_index = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )


        predicted_index = int(
            predicted_index.item()
        )


        predicted_class = (
            self.class_names[
                predicted_index
            ]
        )


        confidence_percent = (
            float(confidence.item())
            * 100
        )


        all_probabilities = {}


        for index, class_name in enumerate(
            self.class_names
        ):

            probability = (
                probabilities[
                    0,
                    index
                ].item()
                * 100
            )

            all_probabilities[
                str(class_name)
            ] = round(
                float(probability),
                2
            )


        result = {

            "prediction":
                str(predicted_class),

            "confidence":
                round(
                    confidence_percent,
                    2
                ),

            "probabilities":
                all_probabilities
        }


        return result


# ============================================================
# GLOBAL MODEL INSTANCE
# ============================================================

model_service = PneumoniaModelService()
