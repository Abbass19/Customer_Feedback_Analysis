"""
Streamlit Customer Feedback Analysis - Simple Working Template
Filename: streamlit_feedback_simple.py

A simplified version that will run without issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import random

# ---------------------- Simple Data Generation ----------------------

def generate_random_feedback():
    """Generate random feedback text for testing"""
    feedback_types = [
        "The product is amazing and works perfectly! I love the new features.",
        "Very disappointed with the customer service. Had to wait 30 minutes on hold.",
        "Could you please add dark mode? That would be really helpful.",
        "The app keeps crashing when I try to upload photos. Very frustrating.",
        "Excellent quality and fast delivery. Will definitely buy again!",
        "The interface is confusing and hard to navigate. Needs improvement.",
        "I would like to request a refund for my last purchase.",
        "Best customer support ever! They solved my issue in minutes.",
        "The shipping was delayed by 3 days without any notification.",
        "Great value for money. Highly recommended to everyone."
    ]
    return random.choice(feedback_types)



# These are good. I take them 0: negative, 1 : positive, 2 neutral
# We take these as well and make a query
def analyze_sentiment_simple(text):
    """Simple sentiment analysis"""
    if not text or text.strip() == "":
        return "neutral", 0.0

    # Simple rule-based
    text_lower = text.lower()
    positive_words = ["amazing", "excellent", "great", "love", "perfectly", "best", "happy", "good", "awesome"]
    negative_words = ["disappointed", "crashing", "frustrating", "confusing", "delayed", "refund", "bad", "broken", "error"]

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return "positive", round(random.uniform(0.3, 0.9), 2)
    elif negative_count > positive_count:
        return "negative", round(random.uniform(-0.9, -0.3), 2)
    else:
        return "neutral", round(random.uniform(-0.2, 0.2), 2)





# The database query is for both
def classify_feedback_simple(text):
    """Simple classification"""
    text_lower = text.lower()
    if any(word in text_lower for word in ["not working", "broken", "error", "complain", "bad", "refund", "angry", "disappointed", "crashing", "frustrating"]):
        return "Complaint"
    if any(word in text_lower for word in ["love", "great", "excellent", "thank", "awesome", "well done", "happy", "amazing", "best", "recommended"]):
        return "Praise"
    if any(word in text_lower for word in ["please", "could you", "i want", "feature", "add", "would be nice", "request", "dark mode"]):
        return "Request"
    return "Other"

def generate_sample_data(num_samples=8):
    """Generate sample data for initial display"""
    sample_data = []
    for i in range(num_samples):
        text = generate_random_feedback()
        sentiment, polarity = analyze_sentiment_simple(text)
        classification = classify_feedback_simple(text)

        # Generate some random entities
        companies = ["TechCorp", "ProductX", "ServicePro", "AppPlus", "CloudSolution"]
        entities = [(random.choice(companies), "PRODUCT") for _ in range(random.randint(1, 2))]

        sample_data.append({
            'id': f"sample_{i}",
            'text': text,
            'entities': entities,
            'sentiment': sentiment,
            'polarity': polarity,
            'classification': classification,
            'timestamp': datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, 72))
        })
    return sample_data

# ---------------------- Initialize Session State ----------------------

if 'records' not in st.session_state:
    st.session_state.records = generate_sample_data()

if 'feedback_text' not in st.session_state:
    st.session_state.feedback_text = ""

# ---------------------- Main App ----------------------

def main():
    st.set_page_config(page_title="Customer Feedback Analysis", layout="wide")

    # Header
    st.title("📊 Customer Feedback Analysis")
    st.markdown("**Template with sample data - Ready for your real data**")

    # Main layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("💬 Input Feedback")

        # Random feedback generator
        if st.button("🎲 Generate Random Feedback", use_container_width=True):
            st.session_state.feedback_text = generate_random_feedback()
            st.rerun()

        # Text input
        feedback_text = st.text_area(
            "Enter customer feedback:",
            value=st.session_state.feedback_text,
            height=150,
            placeholder="Type your feedback here or generate random sample..."
        )

        # Analysis buttons
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("🔍 Analyze Text", use_container_width=True):
                if feedback_text.strip():
                    sentiment, polarity = analyze_sentiment_simple(feedback_text)
                    classification = classify_feedback_simple(feedback_text)

                    st.success("Analysis Complete!")
                    st.write(f"**Sentiment:** {sentiment} (polarity: {polarity:.2f})")
                    st.write(f"**Classification:** {classification}")
                else:
                    st.error("Please enter some text to analyze")

        with col1b:
            if st.button("📥 Submit Feedback", type="primary", use_container_width=True):
                if feedback_text.strip():
                    # Create new record
                    sentiment, polarity = analyze_sentiment_simple(feedback_text)
                    classification = classify_feedback_simple(feedback_text)
                    entities = [(f"Product{random.randint(1, 100)}", "PRODUCT")]

                    new_record = {
                        'id': str(datetime.datetime.now().timestamp()),
                        'text': feedback_text,
                        'entities': entities,
                        'sentiment': sentiment,
                        'polarity': polarity,
                        'classification': classification,
                        'timestamp': datetime.datetime.now()
                    }

                    # Add to records
                    st.session_state.records.insert(0, new_record)
                    st.session_state.feedback_text = ""
                    st.success("Feedback submitted successfully!")
                    st.rerun()
                else:
                    st.error("Please enter some feedback text")

        st.markdown("---")
        st.header("📈 Quick Stats")
        if st.session_state.records:
            df = pd.DataFrame(st.session_state.records)
            sentiment_counts = df['sentiment'].value_counts()
            classification_counts = df['classification'].value_counts()

            st.write(f"**Total Records:** {len(st.session_state.records)}")
            st.write("**Sentiment Distribution:**")
            for sentiment, count in sentiment_counts.items():
                st.write(f"  - {sentiment}: {count}")

            st.write("**Classification Distribution:**")
            for classification, count in classification_counts.items():
                st.write(f"  - {classification}: {count}")
        else:
            st.info("No feedback records yet")

    with col2:
        st.header("📋 Feedback Records")

        if st.session_state.records:
            # Convert to DataFrame for display
            display_data = []
            for record in st.session_state.records:
                display_data.append({
                    'Time': record['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'Feedback': record['text'][:100] + "..." if len(record['text']) > 100 else record['text'],
                    'Sentiment': record['sentiment'],
                    'Type': record['classification'],
                    'Full_Text': record['text']  # Hidden column for full text
                })

            df_display = pd.DataFrame(display_data)

            # Filters
            st.subheader("🔍 Filters")
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment",
                    options=['positive', 'neutral', 'negative'],
                    default=['positive', 'neutral', 'negative']
                )
            with filter_col2:
                type_options = list(set([r['classification'] for r in st.session_state.records]))
                type_filter = st.multiselect(
                    "Filter by Type",
                    options=type_options,
                    default=type_options
                )

            # Apply filters
            filtered_df = df_display[
                (df_display['Sentiment'].isin(sentiment_filter)) &
                (df_display['Type'].isin(type_filter))
            ]

            # This presentation way, I prefer it to be a  scroll bar.
            st.write(f"Showing {len(filtered_df)} of {len(df_display)} records")

            # Display table
            for idx, row in filtered_df.iterrows():
                with st.expander(f"{row['Time']} - {row['Sentiment']} - {row['Type']}"):
                    st.write(row['Full_Text'])
                    sentiment_color = {
                        'positive': '🟢',
                        'negative': '🔴',
                        'neutral': '⚪'
                    }
                    st.write(f"{sentiment_color.get(row['Sentiment'], '⚪')} **{row['Sentiment'].title()}** | 📝 **{row['Type']}**")

            # Export buttons
            st.subheader("📤 Export Data")
            export_col1, export_col2 = st.columns(2)

            with export_col1:
                if st.button("Download CSV", use_container_width=True):
                    csv = pd.DataFrame(st.session_state.records).to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="feedback_data.csv">📥 Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)

            with export_col2:
                if st.button("Clear All Data", use_container_width=True):
                    st.session_state.records = []
                    st.success("All data cleared!")
                    st.rerun()

        else:
            st.info("No feedback records available. Submit some feedback to see them here!")

    # Analytics Section
    st.markdown("---")
    st.header("📊 Analytics Dashboard")

    if st.session_state.records:
        # Prepare data for charts
        df = pd.DataFrame(st.session_state.records)

        # Create charts in columns
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Sentiment pie chart
            sentiment_counts = df['sentiment'].value_counts()
            fig1 = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Sentiment Distribution',
                color=sentiment_counts.index,
                color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            )
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            # Classification bar chart
            classification_counts = df['classification'].value_counts()
            fig2 = px.bar(
                x=classification_counts.index,
                y=classification_counts.values,
                title='Feedback Type Distribution',
                labels={'x': 'Type', 'y': 'Count'},
                color=classification_counts.index
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Timeline chart
        st.subheader("📅 Feedback Over Time")
        if len(st.session_state.records) > 1:
            timeline_data = []
            for record in st.session_state.records:
                timeline_data.append({
                    'date': record['timestamp'].date(),
                    'sentiment': record['sentiment'],
                    'type': record['classification']
                })

            timeline_df = pd.DataFrame(timeline_data)
            daily_counts = timeline_df.groupby(['date', 'sentiment']).size().reset_index(name='count')

            fig3 = px.line(
                daily_counts,
                x='date',
                y='count',
                color='sentiment',
                title='Daily Feedback Trend by Sentiment',
                markers=True
            )
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("Submit some feedback to see analytics here!")

    # Template information
    st.markdown("---")
    st.header("ℹ️ Template Information")

    st.markdown("""
    ### 🎯 This is a working template with sample data
    
    **Next steps to customize:**
    1. **Replace sample data** with your real database queries
    2. **Add your actual NLP models** for sentiment and classification
    3. **Connect to your data sources** (APIs, databases, etc.)
    4. **Customize the analysis** to match your specific needs
    
    **Current features:**
    - ✅ Sample data generation for testing
    - ✅ Basic sentiment analysis
    - ✅ Feedback classification
    - ✅ Data filtering and export
    - ✅ Analytics dashboard
    - ✅ Responsive design
    
    The template is ready - just replace the data generation and analysis functions with your real logic!
    """)

if __name__ == "__main__":
    main()

    # streamlit run app.py