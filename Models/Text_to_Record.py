from Classification_Model.Solution_1_BERT_MultiHead_Classifier.Classification_Sentiment_Analysis.BERT_Classification_Model import predict_feedback_array
from NER_Model.GLiNER_NER_Model import extract_entities_array

def text_to_record(text):
    """
    Convert a single text to a merged record:
    [text, 5 sentiment numbers, 12 NER strings]
    Missing NER values are replaced with empty string.
    """
    # 1️⃣ Feedback predictions (list of 5 numbers)
    feedback_preds = predict_feedback_array(text)

    # 2️⃣ NER predictions (list of 12 strings)
    ner_preds = extract_entities_array(text)

    # Replace None with empty string
    ner_preds_clean = [val if val is not None else " " for val in ner_preds]

    # 3️⃣ Merge everything into a single record
    record = [text] + feedback_preds + ner_preds_clean
    return record

# ----------------------------
# Example usage
# ----------------------------
example_text = "مستشفى كبير اشكر الدكتوره هيبات الصديقي صاحبه الوجه البشوش والتعامل الراقي تشعرك بالراحه والاطمئنان 💖الله يسعدها"
record = text_to_record(example_text)
print(record)

for i in range(len(record)):
    print(f" {i} : {record[i]}")