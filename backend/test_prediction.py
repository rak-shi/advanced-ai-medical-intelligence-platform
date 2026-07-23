from app.services.model_service import model_service
from app.services.gradcam_service import generate_gradcam


IMAGE_PATH = (
    "../dataset/chest_xray/test/PNEUMONIA/"
)

import os


images = [
    file
    for file in os.listdir(IMAGE_PATH)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


if not images:
    raise RuntimeError(
        "No test X-ray images found."
    )


test_image = os.path.join(
    IMAGE_PATH,
    images[0]
)


print("\nTesting image:")
print(test_image)


result = model_service.predict(
    test_image
)


print("\nPrediction Result")

print("--------------------")

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Confidence:",
    result["confidence"],
    "%"
)

print(
    "Probabilities:",
    result["probabilities"]
)


gradcam_result = generate_gradcam(
    test_image
)


print("\nGrad-CAM Result")

print("--------------------")

print(
    "Predicted class:",
    gradcam_result["predicted_class"]
)

print(
    "Grad-CAM image:",
    gradcam_result["gradcam_path"]
)