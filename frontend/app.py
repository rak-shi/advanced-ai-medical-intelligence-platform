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
).rstrip("/")

# Render free instances may take time to wake up.
REQUEST_TIMEOUT = 180


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
        color: #9ca3af;
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

if "analysis_success" not in st.session_state:
    st.session_state.analysis_success = False


# ============================================================
# BACKEND HEALTH CHECK
# ============================================================

def check_backend():
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=20
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


# ============================================================
# X-RAY PREDICTION
# ============================================================

def analyze_xray(uploaded_file):
    """
    Send uploaded X-ray to FastAPI backend.
    """

    try:
        file_bytes = uploaded_file.getvalue()

        files = {
            "file": (
                uploaded_file.name,
                file_bytes,
                uploaded_file.type or "image/jpeg"
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            files=files,
            timeout=REQUEST_TIMEOUT
        )

        # Helpful when checking Streamlit logs
        print(
            "Prediction API status:",
            response.status_code
        )

        print(
            "Prediction API response:",
            response.text
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            try:
                data = response.json()

            except ValueError:

                return None, (
                    "Backend returned HTTP 200 but "
                    "the response was not valid JSON."
                )

            if data is None:

                return None, (
                    "Backend returned HTTP 200 with "
                    "a null response."
                )

            if not isinstance(data, dict):

                return None, (
                    "Unexpected backend response format: "
                    f"{type(data).__name__}. "
                    f"Response: {data}"
                )

            if not data:

                return None, (
                    "Backend returned HTTP 200 with "
                    "an empty JSON object."
                )

            if "prediction" not in data:

                return None, (
                    "Backend response does not contain "
                    f"'prediction'. Response: {data}"
                )

            return data, None

        # ----------------------------------------------------
        # BACKEND ERROR
        # ----------------------------------------------------

        try:
            error_data = response.json()

            if isinstance(error_data, dict):

                error_message = error_data.get(
                    "detail",
                    str(error_data)
                )

            else:

                error_message = str(
                    error_data
                )

        except ValueError:

            error_message = (
                response.text
                or "No error message returned."
            )

        return None, (
            f"Backend returned HTTP "
            f"{response.status_code}: "
            f"{error_message}"
        )

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    except requests.Timeout:

        return None, (
            "Prediction request timed out. "
            "The Render free service may be waking up. "
            "Wait approximately one minute and try again."
        )

    # --------------------------------------------------------
    # CONNECTION ERROR
    # --------------------------------------------------------

    except requests.ConnectionError as error:

        return None, (
            "Could not connect to the backend. "
            f"Details: {error}"
        )

    # --------------------------------------------------------
    # REQUEST ERROR
    # --------------------------------------------------------

    except requests.RequestException as error:

        return None, (
            f"Prediction request failed: {error}"
        )

    # --------------------------------------------------------
    # OTHER ERROR
    # --------------------------------------------------------

    except Exception as error:

        return None, (
            "Unexpected prediction error: "
            f"{type(error).__name__}: {error}"
        )


# ============================================================
# GEMINI REPORT
# ============================================================

def generate_report(prediction_id):

    try:

        response = requests.post(
            f"{API_BASE_URL}/api/generate-report/{prediction_id}",
            timeout=REQUEST_TIMEOUT
        )

        print(
            "Report API status:",
            response.status_code
        )

        if response.status_code == 200:

            try:
                data = response.json()

            except ValueError:

                return None, (
                    "Backend returned an invalid JSON report."
                )

            if not data:

                return None, (
                    "Backend returned an empty report."
                )

            return data, None

        try:

            error_data = response.json()

            if isinstance(
                error_data,
                dict
            ):

                error_message = error_data.get(
                    "detail",
                    str(error_data)
                )

            else:

                error_message = str(
                    error_data
                )

        except ValueError:

            error_message = (
                response.text
                or "Unknown backend error."
            )

        return None, (
            f"Backend returned HTTP "
            f"{response.status_code}: "
            f"{error_message}"
        )

    except requests.Timeout:

        return None, (
            "Gemini report generation timed out. "
            "Please try again."
        )

    except requests.ConnectionError as error:

        return None, (
            f"Unable to connect to backend: {error}"
        )

    except requests.RequestException as error:

        return None, str(error)

    except Exception as error:

        return None, (
            f"{type(error).__name__}: {error}"
        )


# ============================================================
# HISTORY
# ============================================================

def get_history():

    try:

        response = requests.get(
            f"{API_BASE_URL}/api/history",
            timeout=60
        )

        if response.status_code == 200:

            data = response.json()

            if isinstance(
                data,
                list
            ):
                return data

        return []

    except requests.RequestException:

        return []


# ============================================================
# DELETE HISTORY
# ============================================================

def delete_history_record(
    prediction_id
):

    try:

        response = requests.delete(
            f"{API_BASE_URL}/api/history/{prediction_id}",
            timeout=60
        )

        return (
            response.status_code
            == 200
        )

    except requests.RequestException:

        return False


# ============================================================
# FORMAT DATE
# ============================================================

def format_date(
    date_value
):

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

    st.title(
        "🩻 Medical AI"
    )

    st.caption(
        "Advanced AI Medical Intelligence Platform"
    )

    st.divider()

    st.subheader(
        "System"
    )

    backend_online = check_backend()

    if backend_online:

        st.success(
            "Backend Online"
        )

    else:

        st.error(
            "Backend Offline"
        )

    st.write(
        "**Model:** EfficientNet-B0"
    )

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

    st.subheader(
        "Model Performance"
    )

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
        "Educational and research prototype only. "
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
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Deep Learning • Medical Image Analysis •
        Grad-CAM Explainability • Gemini LLM • FastAPI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Important:</strong>
        This application is an educational and research
        prototype. Predictions and AI-generated reports
        must not be interpreted as medical diagnoses or
        used as a substitute for evaluation by qualified
        healthcare professionals.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

analysis_tab, history_tab, about_tab = st.tabs(
    [
        "🩻 X-ray Analysis",
        "📋 Prediction History",
        "ℹ️ About"
    ]
)


# ============================================================
# X-RAY ANALYSIS TAB
# ============================================================

with analysis_tab:

    st.header(
        "Chest X-ray Analysis"
    )

    st.write(
        "Upload a chest X-ray image to run the "
        "EfficientNet-B0 pneumonia classifier."
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
    # UPLOADED IMAGE
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
                    width="stretch"
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
                        f"{uploaded_image.width} "
                        f"× "
                        f"{uploaded_image.height}"
                    )
                )

                st.write(
                    "**Mode:**",
                    uploaded_image.mode
                )

        except Exception as error:

            st.error(
                f"Unable to read image: {error}"
            )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        if st.button(
            "🔬 Analyze X-ray",
            type="primary",
            width="stretch"
        ):

            if not backend_online:

                st.error(
                    "FastAPI backend is offline. "
                    "The Render service may be sleeping. "
                    "Wait a few seconds and refresh the page."
                )

            else:

                with st.spinner(
                    "Analyzing chest X-ray..."
                ):

                    result, error = analyze_xray(
                        uploaded_file
                    )

                if error:

                    st.error(
                        f"Analysis failed: {error}"
                    )

                elif result is not None:

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

                    st.session_state.analysis_success = True

                    st.rerun()

                else:

                    st.error(
                        "Prediction failed because no "
                        "result was returned."
                    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    result = st.session_state.get(
        "prediction_result"
    )

    if (
        st.session_state.get(
            "analysis_success"
        )
        and result
    ):

        st.success(
            "Analysis completed successfully."
        )


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
        # RESULT METRICS
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


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

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
        # GRAD-CAM
        # ====================================================

        st.divider()

        st.header(
            "Explainable AI — Grad-CAM"
        )

        st.info(
            "Grad-CAM highlights image regions that "
            "contributed to the neural network prediction. "
            "Highlighted regions indicate model attention "
            "only and are not proof of pathology."
        )

        original_column, gradcam_column = st.columns(
            2
        )


        # ----------------------------------------------------
        # ORIGINAL IMAGE
        # ----------------------------------------------------

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
                    width="stretch"
                )

            else:

                st.info(
                    "Original image unavailable."
                )


        # ----------------------------------------------------
        # GRAD-CAM IMAGE
        # ----------------------------------------------------

        with gradcam_column:

            st.subheader(
                "Grad-CAM Explanation"
            )

            gradcam_url = result.get(
                "gradcam_url"
            )

            if gradcam_url:

                if (
                    gradcam_url.startswith(
                        "http://"
                    )
                    or
                    gradcam_url.startswith(
                        "https://"
                    )
                ):

                    full_gradcam_url = (
                        gradcam_url
                    )

                else:

                    full_gradcam_url = (
                        f"{API_BASE_URL}"
                        f"{gradcam_url}"
                    )

                try:

                    gradcam_response = requests.get(
                        full_gradcam_url,
                        timeout=60
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
                            width="stretch"
                        )

                    else:

                        st.warning(
                            "Grad-CAM image could not "
                            "be loaded from the backend."
                        )

                except requests.RequestException as error:

                    st.warning(
                        f"Unable to load Grad-CAM: {error}"
                    )

            else:

                st.warning(
                    "No Grad-CAM visualization "
                    "was returned."
                )


        # ====================================================
        # GEMINI REPORT
        # ====================================================

        st.divider()

        st.header(
            "AI-Assisted Medical Report"
        )

        st.write(
            "Generate a structured explanatory report "
            "using Gemini based on the model prediction "
            "and probability scores."
        )

        prediction_id = result.get(
            "id"
        )

        if st.button(
            "✨ Generate AI Report",
            type="primary",
            width="stretch",
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

                    report_result, error = (
                        generate_report(
                            prediction_id
                        )
                    )

                if error:

                    st.error(
                        f"Report generation failed: {error}"
                    )

                elif report_result:

                    st.session_state.medical_report = (
                        report_result.get(
                            "medical_report"
                        )
                    )

                    st.rerun()

                else:

                    st.error(
                        "No report was returned."
                    )


        # ====================================================
        # DISPLAY REPORT
        # ====================================================

        if st.session_state.medical_report:

            st.success(
                "AI-assisted report generated successfully."
            )

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
        "Predictions stored in the backend SQLite database."
    )

    if not backend_online:

        st.warning(
            "Backend is unavailable. "
            "Prediction history cannot be retrieved."
        )

    if st.button(
        "🔄 Refresh History",
        width="content"
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

            history_rows.append(
                {
                    "ID":
                        record.get(
                            "id"
                        ),

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
                            f"{float(record.get('confidence', 0)):.2f}%"
                        ),

                    "Normal":
                        (
                            f"{float(record.get('normal_probability', 0)):.2f}%"
                        ),

                    "Pneumonia":
                        (
                            f"{float(record.get('pneumonia_probability', 0)):.2f}%"
                        ),

                    "Report":
                        (
                            "Generated"
                            if record.get(
                                "medical_report"
                            )
                            else
                            "Not Generated"
                        ),

                    "Date":
                        format_date(
                            record.get(
                                "created_at"
                            )
                        )
                }
            )


        history_dataframe = pd.DataFrame(
            history_rows
        )

        st.dataframe(
            history_dataframe,
            width="stretch",
            hide_index=True
        )


        # ====================================================
        # DETAILS
        # ====================================================

        st.divider()

        st.subheader(
            "View Prediction Details"
        )

        available_ids = [
            record.get("id")
            for record in history
            if record.get("id") is not None
        ]


        if available_ids:

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
                            f"{float(selected_record.get('confidence', 0)):.2f}%"
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


                # ============================================
                # HISTORY PROBABILITIES
                # ============================================

                h1, h2 = st.columns(
                    2
                )

                with h1:

                    st.metric(
                        "NORMAL Probability",
                        (
                            f"{float(selected_record.get('normal_probability', 0)):.2f}%"
                        )
                    )

                with h2:

                    st.metric(
                        "PNEUMONIA Probability",
                        (
                            f"{float(selected_record.get('pneumonia_probability', 0)):.2f}%"
                        )
                    )


                # ============================================
                # SAVED REPORT
                # ============================================

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


                # ============================================
                # DELETE
                # ============================================

                st.divider()

                st.subheader(
                    "Delete Record"
                )

                st.warning(
                    "Deleting a prediction removes it "
                    "from the backend database."
                )

                if st.button(
                    "🗑️ Delete Selected Prediction",
                    key=f"delete_{selected_id}"
                ):

                    deleted = (
                        delete_history_record(
                            selected_id
                        )
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
### Advanced AI Medical Intelligence Platform

This project demonstrates an end-to-end AI pipeline for
chest X-ray image analysis.

### AI Model

The system uses **EfficientNet-B0** with transfer learning
to classify chest X-rays into:

- NORMAL
- PNEUMONIA

### Model Performance

| Metric | Score |
|---|---:|
| Test Accuracy | 86.06% |
| Pneumonia Precision | 84.20% |
| Pneumonia Recall | 95.64% |
| Pneumonia F1 Score | 89.56% |
| Best Validation Accuracy | 93.39% |

### Explainable AI

The platform uses **Grad-CAM** to visualize regions of the
X-ray that influenced the neural network's prediction.

Grad-CAM represents model attention and does not establish
medical causation.

### Generative AI

**Gemini** generates an AI-assisted explanatory report
using the classifier prediction, confidence score and
class probabilities.

### Backend

The backend is implemented using **FastAPI** and provides:

- X-ray prediction
- Probability scores
- Grad-CAM visualization
- Gemini report generation
- Prediction history
- Record deletion

### Database

Prediction information is stored using
**SQLite + SQLAlchemy**.

### Deployment

The architecture uses:

- **Streamlit Cloud** — Frontend
- **Render** — FastAPI backend

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
- Render

### Disclaimer

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
