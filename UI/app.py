# Just Can we Make the Head More Decorative
#



"""
Streamlit Customer Feedback Analyzer - Blueprint UI

Layout:
- Header (top)
- Main: two columns
  - Left column (stacked 3 equal parts):
      1) Text input + Record / Stop / Add buttons
      2) NER grid (13 boxes) + Run NER button
      3) Two pie charts (side-by-side)
  - Right column:
      Feedback records display (scrollable), with Refresh / Download CSV / Clear All buttons horizontally

Notes:
- For microphone recording: the app will try to use streamlit_webrtc.
  Install with: pip install streamlit-webrtc
  If streamlit_webrtc isn't available, Record/Stop act as UI toggles (placeholder).
- Replace stub functions (sentiment/classifier/ner/db) with your real implementations.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import datetime
import base64
import random
import io
import json

# Try to import streamlit-webrtc (optional, best for actual in-browser audio)
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
    import av
    STREAMLIT_WEBRTC_AVAILABLE = True
except Exception:
    STREAMLIT_WEBRTC_AVAILABLE = False

# -------------------------
# ---- PLACEHOLDERS ------
# Replace these with your actual model / DB functions
# -------------------------

def analyze_sentiment_scores(text: str) -> List[int]:
    """
    Stub: return five integer sentiment scores 0..3 or similar.
    Replace with your real model inference that returns a list of 5 ints.
    """
    # Example: return random ints 0..3
    return [random.randint(0, 3) for _ in range(5)]

def classify_feedback_simple(text: str) -> str:
    """Stub classifier for display"""
    return random.choice(["Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"])

# NER extractor stub — replace with your GLiNER wrapper
def extract_entities_from_text(text: str) -> Dict[str, Optional[str]]:
    """
    Return a dict with the 13 NER fields.
    Replace with call to your GLiNER model (extract_entities_array or similar).
    """
    keys = [
        "doctor_name", "staff_role", "hospital_name", "department", "specialty",
        "service_area", "price", "feedback_text", "time_expression", "location",
        "quality_aspect", "issue_type", "treatment_type"
    ]
    # Fake extraction: fill doctor/hospital occasionally
    items = {k: "" for k in keys}
    if "دكت" in text or "doctor" in text.lower():
        items["doctor_name"] = "Dr. Auto"
    if "hospital" in text.lower() or "مستشفى" in text:
        items["hospital_name"] = "Example Hospital"
    # always include the original text field
    items["feedback_text"] = text
    return items

# DB functions (session-state based fallback)
def save_record_to_db(record: Dict[str, Any]):
    """
    Replace with actual DB insertion (insert_feedback or API call).
    For now, we use st.session_state['records'] as the demo store.
    """
    st.session_state.records.insert(0, record)

def get_all_records_from_db() -> List[Dict[str, Any]]:
    """
    Replace with real DB query (get_all_feedback).
    """
    return st.session_state.records

def clear_all_records_db():
    st.session_state.records = []

# -------------------------
# ---- Helper utilities ----
# -------------------------
NER_COLUMNS = [
    "doctor_name", "staff_role", "hospital_name", "department", "specialty",
    "service_area", "price", "feedback_text", "time_expression", "location",
    "quality_aspect", "issue_type", "treatment_type"
]

def df_from_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["id", "text", "classification", "timestamp"])
    # Normalize records to flat DF for display/export
    df = pd.json_normalize(records)
    # ensure timestamp formatting
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def download_link_for_df(df: pd.DataFrame, filename: str = "feedback_data.csv"):
    csv_bytes = to_csv_bytes(df)
    b64 = base64.b64encode(csv_bytes).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV File</a>'
    return href

# -------------------------
# ---- Microphone helper ---
# -------------------------
# If streamlit-webrtc is available, define a simple audio processor to collect audio
if STREAMLIT_WEBRTC_AVAILABLE:
    class AudioRecorder(AudioProcessorBase):
        """
        Collect raw audio frames from the browser and store WAV bytes in session_state on stop.
        NOTE: This is a simple example; you may need to adjust formats/sample rates for your transcription.
        """
        def __init__(self):
            self._frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            # We simply append the frame to store it later
            self._frames.append(frame.to_ndarray().T)  # (samples, channels)
            return frame

        def get_audio_bytes(self) -> Optional[bytes]:
            # Convert stored frames to WAV bytes (very basic)
            try:
                import numpy as np
                import soundfile as sf
            except Exception:
                return None
            if not self._frames:
                return None
            arr = np.concatenate(self._frames, axis=0)
            # arrange channels if needed; here assume arr is shape (samples, channels)
            buf = io.BytesIO()
            sf.write(buf, arr, 16000, format="WAV")  # sample rate guess
            return buf.getvalue()

# -------------------------
# ---- Streamlit UI ------
# -------------------------
def main():
    st.set_page_config(page_title="Patient Feedback Analyzer", layout="wide")

    # Initialize session state stores
    if "records" not in st.session_state:
        st.session_state.records = []
    if "feedback_text" not in st.session_state:
        st.session_state.feedback_text = ""
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "last_audio_bytes" not in st.session_state:
        st.session_state.last_audio_bytes = None
    if "ner_results" not in st.session_state:
        st.session_state.ner_results = {k: "" for k in NER_COLUMNS}
    if "last_sentiment_scores" not in st.session_state:
        st.session_state.last_sentiment_scores = None

    # ---- Header with color and icon ----
    header_html = """
    <div style="background-color:#e74c3c;padding:20px;border-radius:10px;color:white;text-align:center;">
        <h1>🩺 Patient Feedback Analyzer</h1>
        <p style="font-size:16px;">Collect and analyze patient feedback efficiently</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.write("")  # spacing

    # ---- Main area split into two columns ----
    # Wrap columns in colored panels for more life
    col1_html = """
    <div style="background-color:#f0f8ff;padding:10px;border-radius:10px;box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
    """
    col2_html = """
    <div style="background-color:#fff0f5;padding:10px;border-radius:10px;box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
    """

    left_col, right_col = st.columns([1, 1], gap="small")


    # -------------------------
    # LEFT COLUMN - stacked 3 equal parts
    # -------------------------
    with left_col:
        # Use three containers to roughly partition vertical space
        input_block = st.container()
        ner_block = st.container()
        stats_block = st.container()

        # ---- Input block (top left) ----
        with input_block:
            st.subheader("Input Feedback")
            # row: text area (large) and vertical tools on the right inside same block
            input_left, input_right = st.columns([3, 1])

            with input_left:
                txt = st.text_area(
                    "Enter feedback (or record voice):",
                    value=st.session_state.feedback_text,
                    height=180,
                    key="feedback_text_area"
                )

            with input_right:
                st.write("")  # spacer

                st.markdown("**Microphone Controls**")

                # Use three columns for buttons
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

                # --- Custom CSS for buttons ---
                st.markdown("""
                <style>
                .stButton>button {
                    height: 2.8em;
                    width: 100%;
                    border-radius: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    text-align: center;
                }
                .record-button {
                    background-color: #2ecc71 !important;
                    color: white !important;
                }
                .stop-button {
                    background-color: #e74c3c !important;
                    color: white !important;
                }
                .add-button {
                    background-color: #3498db !important;
                    color: white !important;
                }
                </style>
                """, unsafe_allow_html=True)

                # Record button (green)
                if btn_col1.button("🎤 Record", key="record_btn", help="Start recording"):
                    st.session_state.recording = True
                    st.experimental_rerun()
                # Stop button (red)
                if btn_col2.button("⏹ Stop", key="stop_btn", help="Stop recording"):
                    st.session_state.recording = False
                    st.experimental_rerun()
                # Add button (blue)
                if btn_col3.button("➕ Add", key="add_btn", help="Add feedback"):
                    feedback_text = st.session_state.get("feedback_text_area", "").strip()
                    if not feedback_text:
                        st.error("No text to add. Type or record then transcribe.")
                    else:
                        sentiment_scores = analyze_sentiment_scores(feedback_text)
                        st.session_state.last_sentiment_scores = sentiment_scores
                        classification = classify_feedback_simple(feedback_text)
                        ner_result = extract_entities_from_text(feedback_text)
                        st.session_state.ner_results = ner_result
                        record = {
                            "id": str(datetime.datetime.now().timestamp()),
                            "text": feedback_text,
                            "sentiment_scores": sentiment_scores,
                            "classification": classification,
                            "ner": ner_result,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        save_record_to_db(record)
                        st.success("Feedback added.")
                        st.session_state.feedback_text = ""
                        st.session_state.feedback_text_area = ""

        # ---- NER block (middle left) ----
        with ner_block:
            st.subheader("NER Results")
            ner_top_row, ner_middle_row, ner_bottom_row = st.columns([1, 1, 1])
            # Lay out 13 boxes in 3 columns (makes 5 rows of up to 3 columns)
            # We'll create a simple grid generator
            cols = st.columns(3)
            # create list of 13 boxes
            boxes = []
            for i, key in enumerate(NER_COLUMNS):
                col_index = i % 3
                with cols[col_index]:
                    val = st.session_state.ner_results.get(key, "")
                    st.text_input(label=key.replace("_", " ").title(), value=val or "", key=f"ner_{key}")
            st.write("")  # spacing
            # Run NER button at top-right of this block
            run_ner_col1, run_ner_col2 = st.columns([3, 1])
            with run_ner_col2:
                if st.button("Run NER"):
                    # run NER on current text input and populate the fields
                    cur_text = st.session_state.get("feedback_text_area", "")
                    ner_out = extract_entities_from_text(cur_text)
                    st.session_state.ner_results = ner_out
                    # push values to the text inputs
                    for k, v in ner_out.items():
                        st.session_state[f"ner_{k}"] = v or ""
                    st.success("NER completed and fields populated.")

        # ---- Stats block (bottom left) ----
        with stats_block:
            st.subheader("Statistics")
            # Two pie charts side-by-side
            stat_col1, stat_col2 = st.columns(2)
            records = get_all_records_from_db()
            df = df_from_records(records)
            # Chart 1: sentiment distribution
            with stat_col1:
                if not df.empty and "sentiment_scores" in df.columns:
                    # derive a simple overall sentiment from first score for display (demo)
                    overall = df["sentiment_scores"].apply(lambda s: s[0] if isinstance(s, list) and s else None)
                    counts = overall.value_counts().sort_index()
                    fig = {
                        "data": [
                            {"labels": counts.index.astype(str).tolist(), "values": counts.values.tolist(), "type": "pie"}
                        ],
                        "layout": {"title": "Sentiment (score0) distribution"}
                    }
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sentiment data to show.")

            # Chart 2: classification distribution
            with stat_col2:
                if not df.empty and "classification" in df.columns:
                    counts = df["classification"].value_counts()
                    fig = {
                        "data": [
                            {"labels": counts.index.tolist(), "values": counts.values.tolist(), "type": "pie"}
                        ],
                        "layout": {"title": "Classification distribution"}
                    }
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No classification data to show.")

    # -------------------------
    # RIGHT COLUMN - Feedback records + filters
    # -------------------------
    with right_col:
        st.subheader("Feedback Records")

        # Filter panels (multi-select)
        st.markdown("### Filters")
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            sentiment_options = ["Positive", "Negative", "Neutral"]
            selected_sentiments = st.multiselect(
                "Sentiment", options=sentiment_options, default=sentiment_options
            )

        with filter_col2:
            classification_options = [
                "Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"
            ]
            selected_classes = st.multiselect(
                "Classification", options=classification_options, default=classification_options
            )

        # -------------------------
        # Refresh / Download / Clear buttons (keep functionality)
        # -------------------------
        refresh_col, download_col, clear_col = st.columns([1, 1, 1])
        with refresh_col:
            if st.button("Refresh"):
                st.experimental_rerun()
        with download_col:
            if st.button("Download CSV"):
                df_all = df_from_records(get_all_records_from_db())
                if df_all.empty:
                    st.info("No data to download.")
                else:
                    st.markdown(download_link_for_df(df_all), unsafe_allow_html=True)
        with clear_col:
            if st.button("Clear All"):
                clear_all_records_db()
                st.success("All records cleared.")
                st.experimental_rerun()

        # -------------------------
        # Display filtered records (automatic)
        # -------------------------
        records = get_all_records_from_db()
        df_display = df_from_records(records)

        # Filter by sentiment
        if "sentiment_scores" in df_display.columns:
            def sentiment_label(score):
                if score is None:
                    return "Neutral"
                if score > 1:
                    return "Positive"
                elif score == 1:
                    return "Neutral"
                else:
                    return "Negative"

            df_display["sentiment_label"] = df_display["sentiment_scores"].apply(
                lambda s: sentiment_label(s[0]) if isinstance(s, list) and s else "Neutral"
            )
            df_display = df_display[df_display["sentiment_label"].isin(selected_sentiments)]

        # Filter by classification
        if "classification" in df_display.columns:
            df_display = df_display[df_display["classification"].isin(selected_classes)]

        # Display records
        if df_display.empty:
            st.info("No records match the selected filters.")
        else:
            for idx, row in df_display.iterrows():
                time_str = ""
                if "timestamp" in row and pd.notnull(row["timestamp"]):
                    try:
                        ts = pd.to_datetime(row["timestamp"])
                        time_str = ts.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        time_str = str(row["timestamp"])
                title = f"{time_str} - {row.get('classification', '')}"
                with st.expander(title, expanded=False):
                    st.write("**Text:**")
                    st.write(row.get("text", ""))
                    st.write("**Sentiment scores:**")
                    st.write(row.get("sentiment_scores", ""))
                    st.write("**NER (selected fields):**")
                    ner = row.get("ner", {})
                    for k in NER_COLUMNS:
                        v = ner.get(k) if isinstance(ner, dict) else ""
                        if v:
                            st.write(f"- **{k}**: {v}")

        # If there are no records at all
        if not records:
            st.info("No feedback records yet.")


if __name__ == "__main__":
    main()
