from Models.Classification_Model.Solution_1_BERT_MultiHead_Classifier.Classification_Sentiment_Analysis.BERT_Classification_Model import predict_feedback_array
from Models.NER_Model.GLiNER_NER_Model import extract_entities_array
from database import DB_ADD_Record


#Insert
def ADD_Comment_API(text: str):
    """
    Processes a single text with sentiment and NER models
    and inserts it into the PostgreSQL database.
    """
    # 1️⃣ Predict sentiment scores (list of 5 numbers)
    sentiment_preds = predict_feedback_array(text)

    # Map to the database columns
    sentiment_scores = {
        "pricing": sentiment_preds[0],
        "appointments": sentiment_preds[1],
        "staff": sentiment_preds[2],
        "customer_service": sentiment_preds[3],
        "emergency_services": sentiment_preds[4]
    }

    # 2️⃣ Predict NER values (list of 12 strings)
    ner_preds = extract_entities_array(text)

    # Replace None with empty string and map to database columns
    ner_values = {
        "doctor_name": ner_preds[0] or "",
        "staff_role": ner_preds[1] or "",
        "hospital_name": ner_preds[2] or "",
        "department": ner_preds[3] or "",
        "specialty": ner_preds[4] or "",
        "service_area": ner_preds[5] or "",
        "price": ner_preds[6] or "",
        "time_expression": ner_preds[7] or "",
        "location": ner_preds[8] or "",
        "quality_aspect": ner_preds[9] or "",
        "issue_type": ner_preds[10] or "",
        "treatment_type": ner_preds[11] or ""
    }

    # 3️⃣ Insert into database
    DB_ADD_Record(text, sentiment_scores, ner_values)






# Example usage
# ----------------------------
example_text = "                                   الدكتور عباس عصام فاضل شاطر جدا جدا           "
ADD_Comment_API(example_text)
print("Record inserted successfully!")