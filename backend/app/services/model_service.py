import os

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class PneumoniaModelService:

    def __init__(self):

        print(f"Loading model from: {MODEL_PATH}")

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        self.class_names = checkpoint["class_names"]

        self.image_size = checkpoint.get(
            "image_size",
            224
        )

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

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Medical imaging model loaded successfully.")

    def prepare_image(self, image):

        if isinstance(image, str):
            image = Image.open(image)

        image = image.convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        return tensor.to(DEVICE)

    def predict(self, image):

        tensor = self.prepare_image(image)

        with torch.no_grad():

            output = self.model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )

        predicted_index = predicted_index.item()

        predicted_class = self.class_names[
            predicted_index
        ]

        confidence = confidence.item()

        all_probabilities = {
            self.class_names[i]:
                round(
                    probabilities[0][i].item() * 100,
                    2
                )
            for i in range(len(self.class_names))
        }

        return {
            "prediction": predicted_class,

            "confidence": round(
                confidence * 100,
                2
            ),

            "probabilities": all_probabilities
        }


model_service = PneumoniaModelService()