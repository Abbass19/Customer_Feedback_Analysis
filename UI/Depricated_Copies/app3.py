#This was Done on Two Steps:

#Step 1:
#This time I used another prompt. To try to make a balanced looking UI
# Okay. Another thing is that I want to make an overall structure for the work.
# Where the parts we are talking about slide into. Like a blueprint.
# I guess in HTML We used to to the same. I will try my best to make my words clear
# (Length is vertical, width is horizental). My aim is to make the screen balanced on the two sides with good shape and no spaces.
# Let's say I split the header with a length of 3. After that length (Going downward) we split into two vertical strides.
# The left vertical stride is split into 3 parts of length(3-3-3 : meaning the same ) (The first for the box of text to write it has a play
# buttom to use the voice as well) (The second for the boxes of NER, 13 actaully, where they get filled with the data from NER, )
# (The third is for the two statisics circle). However the feedback record takes the whole right vertical thing ogf length of 9.
# Is that clear before you jump into a code



#Step 2:
# From the previous one we still have two things missing. Above the table we should have 2 panels.
# These panels offer options to choose from. From Each one we can choose multiple options.
# The first panel is for positive and negative and neutral. The user can choose any combination of that
# The second panel is for the classification; The user can choose any of [Pricing	Appointments	Medical Staff	Customer Service	Emergency Services]


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
    st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide")
    st.title("Customer Feedback Analyzer")

    # Initialize session state stores
    if "records" not in st.session_state:
        # sample records (empty by default; you can fill first time)
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

    # Header (length ~3)
    st.markdown("## Customer Feedback Analyzer")
    st.markdown("---")

    # Main area split into two columns (left stacked into 3 blocks; right records)
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
                # Record / Stop / Add buttons stacked vertically
                st.write("")  # spacer
                # Microphone controls
                if STREAMLIT_WEBRTC_AVAILABLE:
                    st.write("**Microphone**")
                    # webrtc_streamer returns a ctx with state and processor when configured
                    # we will show a webrtc widget when recording is True
                    if st.button("Record"):
                        st.session_state.recording = True
                        st.experimental_rerun()
                    if st.button("Stop"):
                        # stop: collect audio bytes from the recorder if available
                        st.session_state.recording = False
                        st.experimental_rerun()
                    # Start/webrtc_streamer only when recording True
                    if st.session_state.recording:
                        # Show the audio capture UI widget
                        ctx = webrtc_streamer(
                            key="audio",
                            mode=WebRtcMode.SENDRECV,
                            audio_processor_factory=AudioRecorder,
                            media_stream_constraints={"audio": True, "video": False},
                            async_processing=False,
                        )
                        # If the component created a processor instance, keep reference
                        if ctx and ctx.audio_processor:
                            # store the processor to session state so we can extract bytes on stop
                            st.session_state["audio_processor"] = ctx.audio_processor
                        st.info("Recording... Press Stop when finished.")
                    else:
                        # if previously stored audio processor and audio bytes not yet set, try to get bytes
                        proc = st.session_state.get("audio_processor")
                        if proc:
                            try:
                                audio_bytes = proc.get_audio_bytes()
                            except Exception:
                                audio_bytes = None
                            if audio_bytes:
                                st.session_state.last_audio_bytes = audio_bytes
                                # cleanup stored processor
                            st.session_state.pop("audio_processor", None)
                        if st.session_state.last_audio_bytes:
                            st.write("Recorded audio is available.")
                            st.audio(st.session_state.last_audio_bytes)
                else:
                    # Fallback when streamlit-webrtc not installed
                    st.write("**Microphone**")
                    if st.button("Record (placeholder)"):
                        st.session_state.recording = True
                    if st.button("Stop (placeholder)"):
                        st.session_state.recording = False
                        st.success("Recording stopped (placeholder). Please upload audio file if needed.")

                    st.write("If you want real in-browser recording, install: pip install streamlit-webrtc")

                st.write("---")
                # Add button (submit)
                if st.button("Add"):
                    # On Add: run sentiment, classification, optionally NER, then save
                    feedback_text = st.session_state.get("feedback_text_area", "").strip()
                    if not feedback_text:
                        st.error("No text to add. Type or record then transcribe.")
                    else:
                        # 1) Sentiment scores (5)
                        sentiment_scores = analyze_sentiment_scores(feedback_text)
                        st.session_state.last_sentiment_scores = sentiment_scores
                        # 2) Classification
                        classification = classify_feedback_simple(feedback_text)
                        # 3) NER (we can run the NER extractor here or keep it separate)
                        ner_result = extract_entities_from_text(feedback_text)
                        st.session_state.ner_results = ner_result
                        # 4) Build record and save
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
                        # clear input area
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

        # Refresh / Download / Clear buttons
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

        # Display filtered records
        records = get_all_records_from_db()
        if records:
            df_display = df_from_records(records)

            # Filter by sentiment
            if "sentiment_scores" in df_display.columns:
                # map first score to Positive/Neutral/Negative for demo
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
        else:
            st.info("No feedback records yet.")


if __name__ == "__main__":
    main()
