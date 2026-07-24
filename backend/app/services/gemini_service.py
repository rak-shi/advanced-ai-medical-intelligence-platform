import os

from dotenv import load_dotenv
from google import genai


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(ENV_PATH)

API_KEY = os.getenv("GEMINI_API_KEY")


def get_client():

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    return genai.Client(
        api_key=API_KEY
    )


def generate_medical_report(
    prediction: str,
    confidence: float,
    normal_probability: float,
    pneumonia_probability: float
):

    client = get_client()

    prompt = f"""
You are assisting with an educational medical imaging
AI demonstration.

A deep-learning model analyzed a chest X-ray and produced
the following output:

Predicted class: {prediction}
Model confidence: {confidence:.2f}%
NORMAL probability: {normal_probability:.2f}%
PNEUMONIA probability: {pneumonia_probability:.2f}%

Generate a concise AI-assisted report using exactly these
sections:

1. AI Analysis
2. Findings
3. Interpretation
4. Recommended Next Steps
5. Limitations and Disclaimer

Important rules:

- Do not claim that the patient definitively has pneumonia.
- Clearly distinguish the AI model prediction from a
  clinical diagnosis.
- Do not invent patient history, symptoms, age, laboratory
  findings, or radiographic findings that were not supplied.
- Explain that the model output requires interpretation by
  a qualified healthcare professional.
- Recommend appropriate professional evaluation rather than
  prescribing treatment.
- Mention that Grad-CAM is an explanatory visualization and
  does not prove causation or diagnosis.
- Keep the report professional and concise.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty report."
        )

    return response.text.strip()
