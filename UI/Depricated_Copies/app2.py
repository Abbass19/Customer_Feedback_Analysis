import streamlit as st
import pandas as pd
import datetime
import random
import base64
import plotly.express as px

# ---------------------- Simple Data Generation ----------------------

def analyze_sentiment_simple(text):
    """Simple sentiment analysis"""
    if not text or text.strip() == "":
        return "neutral", 0.0
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

def classify_feedback_simple(text):
    """Simple classification"""
    text_lower = text.lower()
    return random.choice(["Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"])

def generate_sample_data(num_samples=10):
    """Generate sample data for initial display"""
    sample_texts = [
        "Excellent medical care and friendly staff",
        "Long waiting times for appointments",
        "Pricing is reasonable and clear",
        "Emergency service was fast and effective",
        "Customer service was rude at first",
        "Doctors were knowledgeable and helpful",
        "Need better communication about procedures",
        "Appointments scheduling is easy",
        "Billing process could be improved",
        "Overall satisfied with hospital experience"
    ]
    sample_data = []
    for i in range(num_samples):
        text = sample_texts[i % len(sample_texts)]
        sentiment, polarity = analyze_sentiment_simple(text)
        classification = classify_feedback_simple(text)
        entities = [(f"Product{random.randint(1, 100)}", "PRODUCT")]
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
    st.title("📊 Customer Feedback Analysis")
    st.markdown("**Sample template connected to database-ready structure**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("💬 Input Feedback")

        feedback_text = st.text_area(
            "Enter customer feedback:",
            value=st.session_state.feedback_text,
            height=150,
            placeholder="Type your feedback here..."
        )

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
            if st.button("📥 Submit Feedback", use_container_width=True):
                if feedback_text.strip():
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
                    st.session_state.records.insert(0, new_record)
                    st.session_state.feedback_text = ""
                    st.success("Feedback submitted successfully!")
                    st.rerun()
                else:
                    st.error("Please enter some feedback text")

    with col2:
        st.header("📋 Feedback Records")
        if st.session_state.records:
            display_data = []
            for record in st.session_state.records:
                display_data.append({
                    'Time': record['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'Feedback': record['text'],
                    'Sentiment': record['sentiment'],
                    'Type': record['classification'],
                    'Full_Text': record['text']
                })

            df_display = pd.DataFrame(display_data)

            # Filters
            st.subheader("🔍 Filters")
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=['positive', 'neutral', 'negative'],
                default=['positive', 'neutral', 'negative']
            )

            type_filter = st.multiselect(
                "Filter by Type",
                options=['Pricing', 'Appointments', 'Medical Staff', 'Customer Service', 'Emergency Services'],
                default=['Pricing', 'Appointments', 'Medical Staff', 'Customer Service', 'Emergency Services']
            )

            filtered_df = df_display[
                (df_display['Sentiment'].isin(sentiment_filter)) &
                (df_display['Type'].isin(type_filter))
            ]

            st.write(f"Showing {len(filtered_df)} of {len(df_display)} records")
            st.dataframe(filtered_df[['Time', 'Feedback', 'Sentiment', 'Type']], height=400)

            # Export
            st.subheader("📤 Export Data")
            if st.button("Download CSV"):
                csv = pd.DataFrame(st.session_state.records).to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="feedback_data.csv">📥 Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

            if st.button("Clear All Data"):
                st.session_state.records = []
                st.success("All data cleared!")
                st.rerun()

    # Analytics Section
    st.markdown("---")
    st.header("📊 Analytics Dashboard")
    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        cols = st.columns(2)
        for i, col in enumerate(cols):
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title=f"Sentiment Pie {i+1}",
                color=sentiment_counts.index,
                color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            )
            col.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Submit some feedback to see analytics here!")

if __name__ == "__main__":
    main()
