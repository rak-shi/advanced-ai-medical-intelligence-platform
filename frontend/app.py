import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://advanced-medical-ai-backend.onrender.com"
)

REQUEST_TIMEOUT = 120


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advanced AI Medical Intelligence Platform",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
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

    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .disclaimer-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d1d5db;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .report-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
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
    """
    Check whether FastAPI backend is available.
    """

    try:

        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def analyze_xray(uploaded_file):
    """
    Send chest X-ray to FastAPI prediction endpoint.

    Returns (result_dict, error_message). Only one of the two
    will be truthy. A 200 status code is NOT enough on its own —
    Render free-tier cold starts / async races can return a 200
    with an empty or partial body, so the body is validated too.
    """

    try:

        file_bytes = uploaded_file.getvalue()

        files = {
            "file": (
                uploaded_file.name,
                file_bytes,
                uploaded_file.type
                or "application/octet-stream"
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            files=files,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:

            try:
                data = response.json()
            except ValueError:
                return None, (
                    "Backend returned a 200 response that "
                    "was not valid JSON: "
                    f"{response.text[:300]!r}"
                )

            # A truthy-but-incomplete body is the bug we hit:
            # success message fires, results section stays empty.
            if not data:
                return None, (
                    "Backend returned an empty response body "
                    "(status 200). This usually means the "
                    "Render instance was cold-starting, or the "
                    "prediction/DB write hadn't finished before "
                    "the response was sent."
                )

            required_keys = ("prediction", "confidence", "probabilities")
            missing = [k for k in required_keys if k not in data]

            if missing:
                return None, (
                    "Backend response is missing expected "
                    f"field(s) {missing}. Raw response: {data!r}"
                )

            return data, None

        try:
            error_message = response.json().get(
                "detail",
                "Prediction failed."
            )
        except Exception:
            error_message = response.text

        return None, error_message

    except requests.ConnectionError:

        return None, (
            "Unable to connect to the FastAPI backend. "
            "Make sure uvicorn is running on port 8000."
        )

    except requests.Timeout:

        return None, (
            "The prediction request timed out. "
            "Please try again."
        )

    except requests.RequestException as error:

        return None, str(error)


def generate_report(prediction_id):
    """
    Ask backend/Gemini to generate a report.
    """

    try:

        response = requests.post(
            (
                f"{API_BASE_URL}"
                f"/api/generate-report/"
                f"{prediction_id}"
            ),
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:

            try:
                data = response.json()
            except ValueError:
                return None, (
                    "Backend returned a 200 response that "
                    "was not valid JSON: "
                    f"{response.text[:300]!r}"
                )

            if not data or "medical_report" not in data:
                return None, (
                    "Backend response is missing the "
                    f"'medical_report' field. Raw response: {data!r}"
                )

            return data, None

        try:
            error_message = response.json().get(
                "detail",
                "Report generation failed."
            )
        except Exception:
            error_message = response.text

        return None, error_message

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
    """
    Retrieve prediction history.
    """

    try:

        response = requests.get(
            f"{API_BASE_URL}/api/history",
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        return []

    except requests.RequestException:
        return []


def delete_history_record(prediction_id):
    """
    Delete a prediction from SQLite through FastAPI.
    """

    try:

        response = requests.delete(
            (
                f"{API_BASE_URL}"
                f"/api/history/"
                f"{prediction_id}"
            ),
            timeout=30
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def format_date(date_value):

    if not date_value:
        return ""

    try:

        date_object = datetime.fromisoformat(
            date_value.replace(
                "Z",
                "+00:00"
            )
        )

        return date_object.strftime(
            "%d %b %Y, %I:%M %p"
        )

    except Exception:
        return date_value


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

    st.write("**Model:** EfficientNet-B0")

    st.write(
        "**Task:** Pneumonia Detection"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.write(
        "**LLM:** Gemini"
    )

    st.write(
        "**Database:** SQLite"
    )

    st.write(
        "**Backend:** FastAPI"
    )

    st.divider()

    st.subheader("Model Performance")

    st.metric(
        "Test Accuracy",
        "86.06%"
    )

    st.metric(
        "Pneumonia Recall",
        "95.64%"
    )

    st.metric(
        "Pneumonia F1",
        "89.56%"
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
    '<div class="main-title">'
    'Advanced AI Medical Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning • Medical Image Analysis • '
    'Grad-CAM Explainability • Gemini LLM • FastAPI'
    '</div>',
    unsafe_allow_html=True
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
    unsafe_allow_html=True
)


# ============================================================
# MAIN TABS
# ============================================================

analysis_tab, history_tab, about_tab = st.tabs(
    [
        "🩻 X-ray Analysis",
        "📋 Prediction History",
        "ℹ️ About"
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
            "png"
        ],
        help=(
            "Supported formats: "
            "JPG, JPEG and PNG"
        )
    )


    # ========================================================
    # DISPLAY UPLOADED IMAGE
    # ========================================================

    if uploaded_file is not None:

        try:

            uploaded_image = Image.open(
                uploaded_file
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
                    use_container_width=True
                )

            with info_column:

                st.write(
                    "**Filename:**",
                    uploaded_file.name
                )

                st.write(
                    "**Format:**",
                    uploaded_image.format
                )

                st.write(
                    "**Image size:**",
                    (
                        f"{uploaded_image.width}"
                        f" × "
                        f"{uploaded_image.height}"
                    )
                )

                st.write(
                    "**Mode:**",
                    uploaded_image.mode
                )

        except Exception:

            st.error(
                "Unable to read this image."
            )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "🔬 Analyze X-ray",
            type="primary",
            use_container_width=True
        ):

            if not backend_online:

                st.error(
                    "FastAPI backend is offline. "
                    "Start the backend before analyzing."
                )

            else:

                with st.spinner(
                    "Analyzing chest X-ray..."
                ):

                    result, error = analyze_xray(
                        uploaded_file
                    )

                # NOTE: previously this only checked `error`.
                # A 200 response with an empty/partial body made
                # `error` falsy too, so the success toast fired
                # while `result` was None/incomplete and the
                # results section below silently rendered nothing.
                if error:

                    st.error(error)

                elif not result:

                    st.error(
                        "The backend returned no usable "
                        "prediction data. Please try again — "
                        "if this keeps happening, the backend "
                        "may still be cold-starting."
                    )

                else:

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

                    st.success(
                        "Analysis completed successfully."
                    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    result = st.session_state.prediction_result


    if result:

        st.divider()

        st.header(
            "AI Prediction"
        )


        prediction = result.get(
            "prediction",
            "Unknown"
        )

        confidence = float(
            result.get(
                "confidence",
                0
            )
        )

        probabilities = result.get(
            "probabilities",
            {}
        )

        normal_probability = float(
            probabilities.get(
                "NORMAL",
                0
            )
        )

        pneumonia_probability = float(
            probabilities.get(
                "PNEUMONIA",
                0
            )
        )


        # ====================================================
        # METRICS
        # ====================================================

        metric1, metric2, metric3 = st.columns(
            3
        )


        with metric1:

            st.metric(
                "Prediction",
                prediction
            )


        with metric2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        with metric3:

            st.metric(
                "Prediction ID",
                result.get(
                    "id",
                    "-"
                )
            )


        st.subheader(
            "Class Probabilities"
        )


        probability1, probability2 = st.columns(
            2
        )


        with probability1:

            st.metric(
                "NORMAL",
                f"{normal_probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        normal_probability / 100,
                        0.0
                    ),
                    1.0
                )
            )


        with probability2:

            st.metric(
                "PNEUMONIA",
                f"{pneumonia_probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        pneumonia_probability / 100,
                        0.0
                    ),
                    1.0
                )
            )


        # ====================================================
        # X-RAY + GRAD-CAM
        # ====================================================

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


        original_column, gradcam_column = st.columns(
            2
        )


        with original_column:

            st.subheader(
                "Original X-ray"
            )

            if (
                st.session_state.uploaded_image_bytes
                is not None
            ):

                st.image(
                    st.session_state.uploaded_image_bytes,
                    caption=(
                        st.session_state.uploaded_filename
                    ),
                    use_container_width=True
                )


        with gradcam_column:

            st.subheader(
                "Grad-CAM Explanation"
            )

            gradcam_url = result.get(
                "gradcam_url"
            )

            if gradcam_url:

                full_gradcam_url = (
                    f"{API_BASE_URL}"
                    f"{gradcam_url}"
                )

                try:

                    gradcam_response = requests.get(
                        full_gradcam_url,
                        timeout=30
                    )

                    if (
                        gradcam_response.status_code
                        == 200
                    ):

                        st.image(
                            gradcam_response.content,
                            caption=(
                                "Grad-CAM visualization"
                            ),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Unable to load Grad-CAM image."
                        )

                except requests.RequestException:

                    st.warning(
                        "Unable to connect to Grad-CAM endpoint."
                    )

            else:

                st.warning(
                    "No Grad-CAM visualization available."
                )


        # ====================================================
        # AI MEDICAL REPORT
        # ====================================================

        st.divider()

        st.header(
            "AI-Assisted Medical Report"
        )

        st.write(
            "Generate a structured explanatory report "
            "using Gemini based on the model output."
        )


        prediction_id = result.get(
            "id"
        )


        if st.button(
            "✨ Generate AI Report",
            type="primary",
            use_container_width=True,
            key="generate_report"
        ):

            if not prediction_id:

                st.error(
                    "Prediction ID is unavailable."
                )

            else:

                with st.spinner(
                    "Gemini is generating the report..."
                ):

                    report_result, error = generate_report(
                        prediction_id
                    )

                # Same class of bug as analyze_xray — check the
                # payload, not just the error string.
                if error:

                    st.error(error)

                elif not report_result or not report_result.get(
                    "medical_report"
                ):

                    st.error(
                        "The backend returned no report text. "
                        "Please try again."
                    )

                else:

                    st.session_state.medical_report = (
                        report_result.get(
                            "medical_report"
                        )
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
                "and does not constitute a medical diagnosis."
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
        use_container_width=False
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

            history_rows.append({

                "ID":
                    record.get("id"),

                "Filename":
                    record.get(
                        "filename",
                        ""
                    ),

                "Prediction":
                    record.get(
                        "prediction",
                        ""
                    ),

                "Confidence":
                    (
                        f"{record.get('confidence', 0):.2f}%"
                    ),

                "Normal":
                    (
                        f"{record.get('normal_probability', 0):.2f}%"
                    ),

                "Pneumonia":
                    (
                        f"{record.get('pneumonia_probability', 0):.2f}%"
                    ),

                "Report":
                    (
                        "Generated"
                        if record.get(
                            "medical_report"
                        )
                        else "Not Generated"
                    ),

                "Date":
                    format_date(
                        record.get(
                            "created_at"
                        )
                    )
            })


        history_dataframe = pd.DataFrame(
            history_rows
        )


        st.dataframe(
            history_dataframe,
            use_container_width=True,
            hide_index=True
        )


        st.divider()

        st.subheader(
            "View Prediction Details"
        )


        available_ids = [
            record.get("id")
            for record in history
            if record.get("id") is not None
        ]


        selected_id = st.selectbox(
            "Select Prediction ID",
            available_ids
        )


        selected_record = next(
            (
                record
                for record in history
                if record.get("id")
                == selected_id
            ),
            None
        )


        if selected_record:

            detail1, detail2, detail3 = st.columns(
                3
            )


            with detail1:

                st.metric(
                    "Prediction",
                    selected_record.get(
                        "prediction",
                        "-"
                    )
                )


            with detail2:

                st.metric(
                    "Confidence",
                    (
                        f"{selected_record.get('confidence', 0):.2f}%"
                    )
                )


            with detail3:

                st.metric(
                    "Prediction ID",
                    selected_record.get(
                        "id",
                        "-"
                    )
                )


            st.write(
                "**Filename:**",
                selected_record.get(
                    "filename",
                    ""
                )
            )


            st.write(
                "**Created:**",
                format_date(
                    selected_record.get(
                        "created_at"
                    )
                )
            )


            if selected_record.get(
                "medical_report"
            ):

                with st.expander(
                    "View AI-Assisted Report",
                    expanded=False
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
                "from the prediction-history database."
            )


            if st.button(
                "🗑️ Delete Selected Prediction",
                key=(
                    f"delete_"
                    f"{selected_id}"
                )
            ):

                deleted = delete_history_record(
                    selected_id
                )


                if deleted:

                    st.success(
                        "Prediction deleted successfully."
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

The **Advanced AI Medical Intelligence Platform**
is an end-to-end AI application demonstrating the
integration of deep learning, medical image analysis,
explainable AI, large language models, REST APIs,
database management and a web interface.

### Deep Learning Model

The chest X-ray classifier uses **EfficientNet-B0**
with transfer learning.

The model was trained to distinguish between:

- **NORMAL**
- **PNEUMONIA**

The supplied training data was split using a stratified
80/20 training-validation strategy, while the original
test set was kept separate for final evaluation.

### Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 86.06% |
| Pneumonia Precision | 84.20% |
| Pneumonia Recall | 95.64% |
| Pneumonia F1 Score | 89.56% |
| Best Validation Accuracy | 93.39% |

### Explainable AI

**Grad-CAM** is used to visualize regions that influenced
the EfficientNet prediction.

Grad-CAM should be interpreted only as an explanatory
visualization of model attention. It does not establish
pathology or anatomical causation.

### Large Language Model

**Gemini** generates a structured AI-assisted report using
the classifier's prediction and confidence scores.

The LLM does not independently establish a medical
diagnosis.

### Backend

The REST API is implemented using **FastAPI**.

Available functionality includes:

- X-ray prediction
- Grad-CAM generation
- AI-assisted report generation
- Prediction-history retrieval
- Prediction deletion
- Health monitoring

### Database

**SQLite + SQLAlchemy** stores prediction history,
probabilities, Grad-CAM paths and generated reports.

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

This project is an educational and technical demonstration.
It has not undergone the clinical validation, regulatory
review, prospective evaluation, calibration analysis, or
external validation required for deployment as a medical
diagnostic system.
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
