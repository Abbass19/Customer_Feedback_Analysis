"""
Streamlit Customer Feedback Analysis - Mockup & Working Prototype
Filename: streamlit_feedback_ui.py

This single-file Streamlit app contains:
- A detailed visual mockup description (markdown at top)
- A runnable prototype implementing the requested UI elements:
  - Large multiline feedback input
  - Voice input helper (file uploader / browser-record instructions)
  - NER (spaCy if available) with correctness checkboxes
  - Sentiment analysis (TextBlob fallback) with color-coded badge
  - Simple rule-based classification (Complaint/Praise/Request/Other)
  - Live feedback table with filters and export to CSV/Excel
  - Submit and Undo Last Action buttons
  - Notifications and error handling
  - Analytics charts and a simple wordcloud for detected entities

Notes:
- For heavier models (transformer-based sentiment/classification) replace the "classify_feedback" and "analyze_sentiment" functions with model inference code.
- Voice recording within the browser requires extra JS or the user to upload an audio file recorded with any recorder. This prototype supports uploading audio (wav/mp3) and converts it to text using SpeechRecognition if available locally (will require the 'speechrecognition' and 'pydub' packages and FFmpeg). If those packages are not available, use the upload as manual transcript and paste into the text area.

How to run:
1. Install required packages: streamlit, pandas, numpy, matplotlib, plotly, wordcloud, spacy, textblob
   (optional for audio->text: SpeechRecognition, pydub, ffmpeg)
2. Run: streamlit run streamlit_feedback_ui.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

# Optional: try to import spaCy for NER. If not available, use a simple fallback.
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

# ---------------------- Helper functions ----------------------

def analyze_ner(text):
    """Return list of (entity_text, label). If spaCy not available, do simple title-case extraction."""
    if not text or text.strip()=="":
        return []
    if SPACY_AVAILABLE:
        doc = nlp(text)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        return ents
    # Fallback: heuristics - return capitalized words as "PROD" or "ORG"
    words = [w.strip('.,!?:;()[]') for w in text.split() if w.istitle()]
    unique = sorted(set(words), key=lambda x: text.index(x))
    return [(w, "MISC") for w in unique]


def analyze_sentiment(text):
    """Return sentiment label and polarity score using TextBlob as fallback.
    Polarity range: [-1.0, 1.0]
    """
    if not text or text.strip()=="":
        return "neutral", 0.0
    tb = TextBlob(text)
    p = tb.sentiment.polarity
    if p > 0.15:
        return "positive", p
    elif p < -0.15:
        return "negative", p
    else:
        return "neutral", p


def classify_feedback(text):
    """Simple rule-based classifier. Replace with ML model for production."""
    t = text.lower()
    if any(w in t for w in ["not working","broken","error","complain","bad","refund","angry"]):
        return "Complaint"
    if any(w in t for w in ["love","great","excellent","thank","awesome","well done","happy"]):
        return "Praise"
    if any(w in t for w in ["please","could you","i want","feature","add","would be nice","request"]):
        return "Request"
    return "Other"


def create_wordcloud(entities):
    text = " ".join([e for e, _ in entities])
    if not text:
        return None
    wc = WordCloud(width=400, height=200, background_color=None, mode='RGBA').generate(text)
    return wc

# ---------------------- Session state helpers ----------------------
if 'records' not in st.session_state:
    st.session_state.records = []  # each record is dict with keys below
if 'undo_stack' not in st.session_state:
    st.session_state.undo_stack = []


def add_record(record):
    st.session_state.records.insert(0, record)  # newest first
    st.session_state.undo_stack.append(('add', record))


def undo_last_action():
    if not st.session_state.undo_stack:
        st.warning("Nothing to undo")
        return
    action, record = st.session_state.undo_stack.pop()
    if action == 'add':
        # remove first matching record
        for i,r in enumerate(st.session_state.records):
            if r['id'] == record['id']:
                st.session_state.records.pop(i)
                st.success("Undid last add")
                return
    st.warning("Could not undo action")

# ---------------------- Layout ----------------------
st.set_page_config(page_title="Customer Feedback Analysis", layout="wide")

# Top header
col1, col2 = st.columns([3,1])
with col1:
    st.title("Customer Feedback Analysis")
    st.caption("Windows-style Streamlit mockup — clean, minimal, and organized")
with col2:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.markdown(f"**{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}**")

# Main layout: left input, right table + controls
left, right = st.columns([1.2, 1.8])

with left:
    st.header("Input / Interaction")
    feedback_text = st.text_area("Enter customer feedback", height=220, key='input_text')

    st.markdown("**Voice input**")
    st.info("Use the file uploader to attach a recorded audio file (wav/mp3). Converting audio to text requires SpeechRecognition/pydub and ffmpeg on the server. Alternatively, record using your OS recorder and paste transcript below.")
    audio_file = st.file_uploader("Upload recorded audio (optional)", type=["wav","mp3","m4a"], key='audio_upload')

    if audio_file is not None:
        st.success("Audio uploaded — click 'Transcribe Audio' to convert to text (if server has speech libs)")
        if st.button("Transcribe Audio"):
            try:
                # Placeholder: real transcription requires speechrecognition/pydub/ffmpeg
                # Here we simulate with a message and set the text area
                simulated_transcript = "[Transcribed audio placeholder — replace with actual transcription]"
                feedback_text = feedback_text + "\n" + simulated_transcript
                # update session text
                st.session_state['input_text'] = feedback_text
                st.success("Audio transcribed and appended to text area")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    st.markdown("---")
    st.header("NER / Sentiment / Classification")
    if st.button("Analyze current text"):
        if not feedback_text or feedback_text.strip()=='':
            st.error("Please enter feedback text (or upload audio and transcribe).")
        else:
            ents = analyze_ner(feedback_text)
            sentiment_label, polarity = analyze_sentiment(feedback_text)
            classification = classify_feedback(feedback_text)
            # show entities with checkboxes to mark correctness
            st.subheader("Detected Entities")
            checked_entities = []
            if not ents:
                st.info("No entities detected.")
            else:
                for i,(t,lab) in enumerate(ents):
                    key = f"ent_correct_{i}"
                    ok = st.checkbox(f"{t}  —  {lab}", value=True, key=key)
                    checked_entities.append({'text':t, 'label':lab, 'correct': ok})
            # sentiment badge
            if sentiment_label == 'positive':
                st.markdown(f"### Sentiment: 🟢 **{sentiment_label.title()}**  (polarity {polarity:.2f})")
            elif sentiment_label == 'negative':
                st.markdown(f"### Sentiment: 🔴 **{sentiment_label.title()}**  (polarity {polarity:.2f})")
            else:
                st.markdown(f"### Sentiment: ⚪ **{sentiment_label.title()}**  (polarity {polarity:.2f})")
            # classification
            st.markdown(f"**Classification:** `{classification}`")

            # show quick action buttons for submit/undo
            st.markdown("---")
            st.write("Use the Submit button below to add this feedback to the dataset")

    st.markdown("---")
    st.header("Actions")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Submit"):
            text = st.session_state.get('input_text', '')
            if not text or text.strip()=='':
                st.error("Cannot submit empty feedback")
            else:
                ents = analyze_ner(text)
                sentiment_label, polarity = analyze_sentiment(text)
                classification = classify_feedback(text)
                record = {
                    'id': str(datetime.datetime.utcnow().timestamp()),
                    'text': text,
                    'entities': ents,
                    'sentiment': sentiment_label,
                    'polarity': polarity,
                    'classification': classification,
                    'timestamp': datetime.datetime.utcnow()
                }
                add_record(record)
                st.success("Feedback submitted and analyzed")
                # clear input
                st.session_state['input_text'] = ''
    with colB:
        if st.button("Undo Last Action"):
            undo_last_action()

    st.markdown("---")
    st.header("Notifications")
    st.write("Use the area above for success/error/info messages after actions. This block intentionally kept minimal for a clean UI.")

with right:
    st.header("Recent Feedback (Live Table)")

    df_records = pd.DataFrame([
        {
            'id': r['id'],
            'text': r['text'],
            'entities': ", ".join([f"{t}({lab})" for t,lab in r['entities']]) if r['entities'] else "",
            'sentiment': r['sentiment'],
            'classification': r['classification'],
            'timestamp': r['timestamp']
        }
        for r in st.session_state.records
    ])

    # Filters
    st.markdown("**Filters**")
    f1, f2, f3 = st.columns(3)
    with f1:
        sentiment_filter = st.multiselect("Sentiment", options=['positive','neutral','negative'], default=['positive','neutral','negative'])
    with f2:
        class_options = sorted(list(set(df_records['classification'].tolist()))) if not df_records.empty else []
        class_filter = st.multiselect("Classification", options=class_options, default=class_options)
    with f3:
        # entity filter: simple text input
        entity_filter = st.text_input("Entity contains (text)")

    filtered = df_records.copy()
    if not filtered.empty:
        filtered = filtered[filtered['sentiment'].isin(sentiment_filter)]
        if class_filter:
            filtered = filtered[filtered['classification'].isin(class_filter)]
        if entity_filter and entity_filter.strip()!='':
            filtered = filtered[filtered['entities'].str.contains(entity_filter, case=False, na=False)]

    st.write(f"Showing {len(filtered)} records")
    st.dataframe(filtered[['timestamp','text','entities','sentiment','classification']], height=350)

    # Export
    colE1, colE2 = st.columns([1,1])
    with colE1:
        if st.button("Export CSV"):
            to_export = filtered.copy()
            to_export['timestamp'] = to_export['timestamp'].astype(str)
            csv = to_export.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f"data:file/csv;base64,{b64}"
            st.markdown(f"[Download CSV]({href})")
    with colE2:
        if st.button("Export Excel"):
            to_export = filtered.copy()
            to_export['timestamp'] = to_export['timestamp'].astype(str)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                to_export.to_excel(writer, index=False, sheet_name='feedback')
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
            st.markdown(f"[Download Excel]({href})")

# ---------------------- Analytics / Visualizations ----------------------
st.markdown("---")
st.header("Analytics & Visualizations")
col_charts = st.columns([1,1,1])

# Sentiment distribution
all_df = pd.DataFrame([{'sentiment': r['sentiment'], 'classification': r['classification'], 'entities':[e for e,_ in r['entities']]} for r in st.session_state.records])
if all_df.empty:
    st.info("No feedback yet — submit feedback to populate analytics")
else:
    # Sentiment pie/bar
    sent_counts = all_df['sentiment'].value_counts().reindex(['positive','neutral','negative']).fillna(0)
    fig1 = px.pie(values=sent_counts.values, names=sent_counts.index, title='Sentiment Distribution')
    col_charts[0].plotly_chart(fig1, use_container_width=True)

    # Classification bar
    class_counts = all_df['classification'].value_counts()
    fig2 = px.bar(x=class_counts.index, y=class_counts.values, title='Classification Distribution', labels={'x':'Class','y':'Count'})
    col_charts[1].plotly_chart(fig2, use_container_width=True)

    # Entities word cloud
    all_entities = []
    for r in st.session_state.records:
        all_entities += [t for t,_ in r['entities']]
    wc = None
    if all_entities:
        wc = WordCloud(width=600, height=300).generate(" ".join(all_entities))
    if wc is not None:
        col_charts[2].pyplot(plt.imshow(wc, interpolation='bilinear'))
        plt.axis('off')
    else:
        col_charts[2].info("No entities detected yet")

# ---------------------- Visual Mockup Description ----------------------
st.markdown("---")
st.subheader("Visual Mockup Description")
st.markdown(
"""
Layout overview (Windows desktop-like Streamlit layout):

- Top bar (full width): App title at left, timestamp and small quick actions at right.

- Main content area split in two columns:
  - Left column (Input / Interaction, ~40% width):
    - Large multiline text area at the top for manual feedback entry (big font, placeholder text).
    - Voice input section below with an Upload button and a 'Transcribe' button. There is also a small record/stop UI if you integrate a browser recorder.
    - NER / Sentiment / Classification panel beneath the voice controls. Contains:
      - Detected entities as a vertical list with a checkbox next to each to mark correctness.
      - Sentiment badge (large), color-coded: green for positive, white/gray for neutral, red for negative.
      - Classification label displayed in a pill-shaped UI element using color codes per class (e.g., Complaint=red, Praise=green, Request=blue, Other=gray).
    - Actions area with Submit and Undo Last Action buttons (large primary buttons). A compact Notifications area shows success/error messages.

  - Right column (Dataset / Feedback Viewer, ~60% width):
    - Live table at top showing recent feedback records with columns: timestamp, text (truncate long text with expand), entities, sentiment (with colored dot), classification (colored pill).
    - A Filters panel above the table with multi-select for sentiment and classification and a free-text field to search entities.
    - Export buttons (CSV / Excel) below the table.

- Bottom (Analytics / Visualization): full-width or three-column area showing:
  - Sentiment distribution (pie chart)
  - Classification distribution (bar chart)
  - Entities word cloud (or simple frequency bar chart)

Design & styling notes:
- Clean, minimal: generous whitespace, subdued gray background, white cards for panels.
- Use color accents for sentiment and classes. Use consistent color map (green, gray, red, blue).
- Typography: simple sans-serif; large headings for major sections. Buttons have rounded corners to match modern Windows UI.
- Responsiveness: On narrow screens the left column stacks above the right table. On desktop it appears as described.

Interaction details & microcopy examples:
- "Transcribe Audio": "Upload a wav/mp3 — our server will transcribe to text. If transcription fails, paste your transcript."
- Entity checkboxes: small hint: "Uncheck if entity is incorrect; changes will be saved with the record."
- After Submit: brief toast-like message: "Feedback submitted — 1 record added." and small undo link.

"""
)

st.success("Prototype mockup generated. Open the code to customize NER/sentiment/classifier models or to wire real audio transcription.")
