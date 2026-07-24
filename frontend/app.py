import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Use the deployed Render backend directly.
# This avoids an old/incorrect API_BASE_URL environment variable
# overriding the deployed backend URL on Streamlit Cloud.
API_BASE_URL = "https://advanced-medical-ai-backend.onrender.com"

REQUEST_TIMEOUT = 180


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advanced AI Medical Intelligence Platform",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .disclaimer-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d1d5db;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "medical_report" not in st.session_state:
    st.session_state.medical_report = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_backend():
    """Check whether the deployed FastAPI backend is available."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=30,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def analyze_xray(uploaded_file):
    """
    Send the uploaded chest X-ray to the deployed FastAPI
    prediction endpoint.
    """

    try:
        file_bytes = uploaded_file.getvalue()

        files = {
            "file": (
                uploaded_file.name,
                file_bytes,
                uploaded_file.type or "image/jpeg",
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            files=files,
            timeout=REQUEST_TIMEOUT,
        )

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        if response.status_code != 200:

            try:
                error_data = response.json()

                if isinstance(error_data, dict):
                    error_message = error_data.get(
                        "detail",
                        error_data,
                    )
                else:
                    error_message = error_data

            except ValueError:
                error_message = response.text

            return None, (
                f"Backend returned HTTP "
                f"{response.status_code}: {error_message}"
            )

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        try:
            data = response.json()

        except ValueError:

            return None, (
                "The backend returned HTTP 200, but the "
                "response was not valid JSON."
            )

        # ----------------------------------------------------
        # VALIDATE RESPONSE
        # ----------------------------------------------------

        if not isinstance(data, dict):

            return None, (
                "The backend response is not a JSON object."
            )

        # Do NOT treat an ERROR object as successful analysis.
        if "ERROR" in data:

            error_data = data["ERROR"]

            if isinstance(error_data, dict):
                error_message = error_data.get(
                    "message",
                    str(error_data),
                )
            else:
                error_message = str(error_data)

            return None, (
                f"Backend prediction error: {error_message}"
            )

        # A successful prediction should contain this field.
        if "prediction" not in data:

            return None, (
                "Unexpected backend response. "
                f"Received: {data}"
            )

        return data, None

    except requests.ConnectionError:

        return None, (
            "Unable to connect to the Render FastAPI backend."
        )

    except requests.Timeout:

        return None, (
            "The prediction request timed out. "
            "The Render service may still be starting. "
            "Please wait and try again."
        )

    except requests.RequestException as error:

        return None, (
            f"Prediction request failed: {error}"
        )

    except Exception as error:

        return None, (
            f"Unexpected prediction error: {error}"
        )


def generate_report(prediction_id):
    """Generate Gemini AI-assisted report."""

    try:

        response = requests.post(
            f"{API_BASE_URL}/api/generate-report/{prediction_id}",
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:

            try:
                data = response.json()

                if isinstance(data, dict):
                    error_message = data.get(
                        "detail",
                        data,
                    )
                else:
                    error_message = data

            except ValueError:
                error_message = response.text

            return None, (
                f"Report generation failed "
                f"(HTTP {response.status_code}): "
                f"{error_message}"
            )

        try:
            data = response.json()
        except ValueError:
            return None, (
                "Report endpoint returned invalid JSON."
            )

        if isinstance(data, dict) and "ERROR" in data:

            error_data = data["ERROR"]

            if isinstance(error_data, dict):
                error_message = error_data.get(
                    "message",
                    str(error_data),
                )
            else:
                error_message = str(error_data)

            return None, error_message

        return data, None

    except requests.ConnectionError:

        return None, (
            "Unable to connect to the backend."
        )

    except requests.Timeout:

        return None, (
            "Gemini report generation timed out."
        )

    except requests.RequestException as error:

        return None, str(error)


def get_history():
    """Retrieve prediction history."""

    try:

        response = requests.get(
            f"{API_BASE_URL}/api/history",
            timeout=30,
        )

        if response.status_code == 200:

            data = response.json()

            if isinstance(data, list):
                return data

            if isinstance(data, dict):

                if "ERROR" in data:
                    return []

                return data.get(
                    "history",
                    data.get(
                        "predictions",
                        [],
                    ),
                )

        return []

    except requests.RequestException:
        return []

    except Exception:
        return []


def delete_history_record(prediction_id):
    """Delete prediction from history."""

    try:

        response = requests.delete(
            f"{API_BASE_URL}/api/history/{prediction_id}",
            timeout=30,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def format_date(date_value):

    if not date_value:
        return ""

    try:

        date_object = datetime.fromisoformat(
            str(date_value).replace(
                "Z",
                "+00:00",
            )
        )

        return date_object.strftime(
            "%d %b %Y, %I:%M %p"
        )

    except Exception:
        return str(date_value)


def safe_float(value):
    """Convert API values safely to float."""

    try:
        return float(value or 0)

    except (ValueError, TypeError):
        return 0.0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🩻 Medical AI")

    st.caption(
        "Advanced AI Medical Intelligence Platform"
    )

    st.divider()

    st.subheader("System")

    backend_online = check_backend()

    if backend_online:
        st.success("Backend Online")
    else:
        st.error("Backend Offline")

    # Temporary diagnostic.
    # Remove this after deployment is confirmed.
    st.caption(
        f"API: {API_BASE_URL}"
    )

    st.write("**Model:** EfficientNet-B0")
    st.write("**Task:** Pneumonia Detection")
    st.write("**Explainability:** Grad-CAM")
    st.write("**LLM:** Gemini")
    st.write("**Database:** SQLite")
    st.write("**Backend:** FastAPI")

    st.divider()

    st.subheader("Model Performance")

    st.metric(
        "Test Accuracy",
        "86.06%",
    )

    st.metric(
        "Pneumonia Recall",
        "95.64%",
    )

    st.metric(
        "Pneumonia F1",
        "89.56%",
    )

    st.divider()

    st.warning(
        "Research/Educational prototype only. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        Advanced AI Medical Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Deep Learning • Medical Image Analysis •
        Grad-CAM Explainability • Gemini LLM • FastAPI
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Important:</strong>
        This application is an educational and research
        prototype. Predictions and AI-generated reports must
        not be interpreted as medical diagnoses or used as a
        substitute for evaluation by qualified healthcare
        professionals.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TABS
# ============================================================

analysis_tab, history_tab, about_tab = st.tabs(
    [
        "🩻 X-ray Analysis",
        "📋 Prediction History",
        "ℹ️ About",
    ]
)


# ============================================================
# ANALYSIS TAB
# ============================================================

with analysis_tab:

    st.header(
        "Chest X-ray Analysis"
    )

    st.write(
        "Upload a chest X-ray image to run "
        "the EfficientNet-B0 pneumonia classifier."
    )

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        help=(
            "Supported formats: JPG, JPEG and PNG"
        ),
    )


    # ========================================================
    # UPLOADED IMAGE
    # ========================================================

    if uploaded_file is not None:

        try:

            image_bytes = uploaded_file.getvalue()

            uploaded_image = Image.open(
                BytesIO(image_bytes)
            )

            st.subheader(
                "Uploaded X-ray"
            )

            image_column, info_column = st.columns(
                [1, 1]
            )

            with image_column:

                st.image(
                    uploaded_image,
                    caption=uploaded_file.name,
                    use_container_width=True,
                )

            with info_column:

                st.write(
                    "**Filename:**",
                    uploaded_file.name,
                )

                st.write(
                    "**Format:**",
                    uploaded_image.format,
                )

                st.write(
                    "**Image size:**",
                    (
                        f"{uploaded_image.width} × "
                        f"{uploaded_image.height}"
                    ),
                )

                st.write(
                    "**Mode:**",
                    uploaded_image.mode,
                )

        except Exception as error:

            st.error(
                f"Unable to read this image: {error}"
            )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "🔬 Analyze X-ray",
            type="primary",
            use_container_width=True,
        ):

            # Clear previous result.
            st.session_state.prediction_result = None
            st.session_state.medical_report = None

            if not backend_online:

                st.error(
                    "FastAPI backend is offline."
                )

            else:

                with st.spinner(
                    "Analyzing chest X-ray..."
                ):

                    result, error = analyze_xray(
                        uploaded_file
                    )

                if error:

                    st.error(error)

                else:

                    st.success(
                        "Analysis completed successfully."
                    )

                    st.session_state.prediction_result = (
                        result
                    )

                    st.session_state.uploaded_image_bytes = (
                        uploaded_file.getvalue()
                    )

                    st.session_state.uploaded_filename = (
                        uploaded_file.name
                    )

                    st.session_state.medical_report = None


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    result = st.session_state.prediction_result

    if result is not None:

        st.divider()

        st.header(
            "AI Prediction"
        )

        if not isinstance(result, dict):

            st.error(
                "The backend response is not "
                "a JSON object."
            )

        elif "prediction" not in result:

            st.error(
                "Prediction information is missing "
                "from the backend response."
            )

        else:

            # =================================================
            # READ BACKEND VALUES
            # =================================================

            prediction = result.get(
                "prediction",
                "Unknown",
            )

            confidence = safe_float(
                result.get(
                    "confidence",
                    0,
                )
            )

            probabilities = result.get(
                "probabilities",
                {},
            )

            if not isinstance(
                probabilities,
                dict,
            ):
                probabilities = {}

            normal_probability = safe_float(
                probabilities.get(
                    "NORMAL",
                    probabilities.get(
                        "normal",
                        result.get(
                            "normal_probability",
                            0,
                        ),
                    ),
                )
            )

            pneumonia_probability = safe_float(
                probabilities.get(
                    "PNEUMONIA",
                    probabilities.get(
                        "pneumonia",
                        result.get(
                            "pneumonia_probability",
                            0,
                        ),
                    ),
                )
            )

            prediction_id = result.get(
                "id",
                result.get(
                    "prediction_id"
                ),
            )


            # =================================================
            # METRICS
            # =================================================

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Prediction",
                    str(prediction),
                )

            with metric2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%",
                )

            with metric3:

                st.metric(
                    "Prediction ID",
                    (
                        prediction_id
                        if prediction_id is not None
                        else "-"
                    ),
                )


            # =================================================
            # PROBABILITIES
            # =================================================

            st.subheader(
                "Class Probabilities"
            )

            probability1, probability2 = st.columns(2)

            with probability1:

                st.metric(
                    "NORMAL",
                    f"{normal_probability:.2f}%",
                )

                st.progress(
                    min(
                        max(
                            normal_probability / 100,
                            0.0,
                        ),
                        1.0,
                    )
                )

            with probability2:

                st.metric(
                    "PNEUMONIA",
                    f"{pneumonia_probability:.2f}%",
                )

                st.progress(
                    min(
                        max(
                            pneumonia_probability / 100,
                            0.0,
                        ),
                        1.0,
                    )
                )


            # =================================================
            # GRAD-CAM
            # =================================================

            st.divider()

            st.header(
                "Explainable AI — Grad-CAM"
            )

            st.write(
                "Grad-CAM highlights image regions that "
                "contributed to the neural network's "
                "prediction. Highlighted regions are not "
                "proof of pathology."
            )

            original_column, gradcam_column = st.columns(2)

            with original_column:

                st.subheader(
                    "Original X-ray"
                )

                if (
                    st.session_state.uploaded_image_bytes
                    is not None
                ):

                    try:

                        original_image = Image.open(
                            BytesIO(
                                st.session_state
                                .uploaded_image_bytes
                            )
                        )

                        st.image(
                            original_image,
                            caption=(
                                st.session_state
                                .uploaded_filename
                            ),
                            use_container_width=True,
                        )

                    except Exception as error:

                        st.warning(
                            "Unable to display original "
                            f"image: {error}"
                        )

            with gradcam_column:

                st.subheader(
                    "Grad-CAM Explanation"
                )

                gradcam_url = result.get(
                    "gradcam_url",
                    result.get(
                        "gradcam"
                    ),
                )

                if gradcam_url:

                    gradcam_url = str(
                        gradcam_url
                    ).strip()

                    if gradcam_url.startswith(
                        (
                            "http://",
                            "https://",
                        )
                    ):

                        full_gradcam_url = gradcam_url

                    else:

                        if not gradcam_url.startswith("/"):
                            gradcam_url = "/" + gradcam_url

                        full_gradcam_url = (
                            f"{API_BASE_URL}"
                            f"{gradcam_url}"
                        )

                    try:

                        gradcam_response = requests.get(
                            full_gradcam_url,
                            timeout=60,
                        )

                        if (
                            gradcam_response.status_code
                            == 200
                        ):

                            try:

                                gradcam_image = Image.open(
                                    BytesIO(
                                        gradcam_response.content
                                    )
                                )

                                st.image(
                                    gradcam_image,
                                    caption=(
                                        "Grad-CAM visualization"
                                    ),
                                    use_container_width=True,
                                )

                            except Exception as error:

                                st.warning(
                                    "Grad-CAM endpoint responded, "
                                    "but the returned file could "
                                    "not be decoded as an image: "
                                    f"{error}"
                                )

                        else:

                            st.warning(
                                "Unable to load Grad-CAM "
                                "image. HTTP status: "
                                f"{gradcam_response.status_code}"
                            )

                    except requests.RequestException as error:

                        st.warning(
                            "Unable to connect to Grad-CAM "
                            f"endpoint: {error}"
                        )

                else:

                    st.warning(
                        "No Grad-CAM visualization available."
                    )


            # =================================================
            # AI REPORT
            # =================================================

            st.divider()

            st.header(
                "AI-Assisted Medical Report"
            )

            st.write(
                "Generate a structured explanatory report "
                "using Gemini based on the model output."
            )

            if st.button(
                "✨ Generate AI Report",
                type="primary",
                use_container_width=True,
                key="generate_report",
            ):

                if prediction_id is None:

                    st.error(
                        "Prediction ID is unavailable."
                    )

                else:

                    with st.spinner(
                        "Gemini is generating the report..."
                    ):

                        (
                            report_result,
                            report_error,
                        ) = generate_report(
                            prediction_id
                        )

                    if report_error:

                        st.error(
                            report_error
                        )

                    else:

                        if isinstance(
                            report_result,
                            dict,
                        ):

                            st.session_state.medical_report = (
                                report_result.get(
                                    "medical_report",
                                    report_result.get(
                                        "report"
                                    ),
                                )
                            )

                        else:

                            st.session_state.medical_report = (
                                str(report_result)
                            )

                        st.success(
                            "AI-assisted report generated."
                        )

            if st.session_state.medical_report:

                st.markdown(
                    st.session_state.medical_report
                )

                st.warning(
                    "This AI-assisted report is for "
                    "educational and research purposes only "
                    "and does not constitute a medical "
                    "diagnosis."
                )


# ============================================================
# HISTORY TAB
# ============================================================

with history_tab:

    st.header(
        "Prediction History"
    )

    st.write(
        "Predictions stored in the SQLite database."
    )

    if st.button(
        "🔄 Refresh History",
    ):

        st.rerun()

    history = get_history()

    if not history:

        st.info(
            "No prediction history is currently available."
        )

    else:

        history_rows = []

        for record in history:

            if not isinstance(
                record,
                dict,
            ):
                continue

            history_confidence = safe_float(
                record.get(
                    "confidence",
                    0,
                )
            )

            history_normal = safe_float(
                record.get(
                    "normal_probability",
                    0,
                )
            )

            history_pneumonia = safe_float(
                record.get(
                    "pneumonia_probability",
                    0,
                )
            )

            history_rows.append(
                {
                    "ID": record.get("id"),

                    "Filename": record.get(
                        "filename",
                        "",
                    ),

                    "Prediction": record.get(
                        "prediction",
                        "",
                    ),

                    "Confidence": (
                        f"{history_confidence:.2f}%"
                    ),

                    "Normal": (
                        f"{history_normal:.2f}%"
                    ),

                    "Pneumonia": (
                        f"{history_pneumonia:.2f}%"
                    ),

                    "Report": (
                        "Generated"
                        if record.get(
                            "medical_report"
                        )
                        else "Not Generated"
                    ),

                    "Date": format_date(
                        record.get(
                            "created_at"
                        )
                    ),
                }
            )

        if history_rows:

            history_dataframe = pd.DataFrame(
                history_rows
            )

            st.dataframe(
                history_dataframe,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.subheader(
                "View Prediction Details"
            )

            available_ids = [
                record.get("id")
                for record in history
                if isinstance(record, dict)
                and record.get("id") is not None
            ]

            if available_ids:

                selected_id = st.selectbox(
                    "Select Prediction ID",
                    available_ids,
                )

                selected_record = next(
                    (
                        record
                        for record in history
                        if isinstance(record, dict)
                        and record.get("id")
                        == selected_id
                    ),
                    None,
                )

                if selected_record:

                    (
                        detail1,
                        detail2,
                        detail3,
                    ) = st.columns(3)

                    with detail1:

                        st.metric(
                            "Prediction",
                            selected_record.get(
                                "prediction",
                                "-",
                            ),
                        )

                    with detail2:

                        selected_confidence = safe_float(
                            selected_record.get(
                                "confidence",
                                0,
                            )
                        )

                        st.metric(
                            "Confidence",
                            (
                                f"{selected_confidence:.2f}%"
                            ),
                        )

                    with detail3:

                        st.metric(
                            "Prediction ID",
                            selected_record.get(
                                "id",
                                "-",
                            ),
                        )

                    st.write(
                        "**Filename:**",
                        selected_record.get(
                            "filename",
                            "",
                        ),
                    )

                    st.write(
                        "**Created:**",
                        format_date(
                            selected_record.get(
                                "created_at"
                            )
                        ),
                    )

                    if selected_record.get(
                        "medical_report"
                    ):

                        with st.expander(
                            "View AI-Assisted Report"
                        ):

                            st.markdown(
                                selected_record[
                                    "medical_report"
                                ]
                            )

                    st.divider()

                    st.subheader(
                        "Delete Record"
                    )

                    st.warning(
                        "Deleting a record removes it "
                        "from prediction history."
                    )

                    if st.button(
                        "🗑️ Delete Selected Prediction",
                        key=f"delete_{selected_id}",
                    ):

                        deleted = delete_history_record(
                            selected_id
                        )

                        if deleted:

                            st.success(
                                "Prediction deleted "
                                "successfully."
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Unable to delete prediction."
                            )


# ============================================================
# ABOUT TAB
# ============================================================

with about_tab:

    st.header(
        "About the Platform"
    )

    st.markdown(
        """
### Project Overview

The **Advanced AI Medical Intelligence Platform** is an
end-to-end AI application demonstrating deep learning,
medical image analysis, explainable AI, large language
models, REST APIs, database management and a web interface.

### Deep Learning Model

The chest X-ray classifier uses **EfficientNet-B0**
with transfer learning.

The model classifies:

- **NORMAL**
- **PNEUMONIA**

### Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 86.06% |
| Pneumonia Precision | 84.20% |
| Pneumonia Recall | 95.64% |
| Pneumonia F1 Score | 89.56% |
| Best Validation Accuracy | 93.39% |

### Explainable AI

**Grad-CAM** visualizes regions that influenced the
EfficientNet prediction.

### Large Language Model

**Gemini** generates a structured AI-assisted report using
the classifier prediction and confidence scores.

### Backend

The REST API is implemented using **FastAPI**.

Functionality includes:

- X-ray prediction
- Grad-CAM generation
- Gemini report generation
- Prediction history
- Prediction deletion
- Health monitoring

### Database

**SQLite + SQLAlchemy** stores prediction history and
generated reports.

### Technology Stack

- Python
- PyTorch
- torchvision
- EfficientNet-B0
- Grad-CAM
- Gemini
- FastAPI
- SQLAlchemy
- SQLite
- Streamlit
- Pandas

### Important Limitation

This application is an educational and research prototype.
It has not undergone the clinical validation or regulatory
review required for use as a medical diagnostic system.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Advanced AI Medical Intelligence Platform • "
    "Educational & Research Prototype"
)
