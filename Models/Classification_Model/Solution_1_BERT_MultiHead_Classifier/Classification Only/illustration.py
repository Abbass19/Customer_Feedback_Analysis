# GLiNER_NER_Model.py

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


# 🔹 Define the model class (must match the trained one)
class MultiHeadBinaryClassifier(nn.Module):
    def __init__(self, bert_model_name="aubmindlab/bert-base-arabertv2", num_heads=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_heads)])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = [head(cls_embedding).squeeze(-1) for head in self.heads]
        return torch.stack(logits, dim=1)


# 🔹 Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load model (state_dict version)
model = MultiHeadBinaryClassifier()
model.load_state_dict(torch.load("5_Head_Classification_Only.pt", map_location=device))
model.to(device)
model.eval()

# 🔹 Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")


# 🔹 Function to predict on a single sentence
def predict_feedback(text):
    encodings = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)  # Since original is binary classification
    return probs.cpu().numpy()


# 🔹 Example usage
if __name__ == "__main__":
    sentence = "شكرا لهم فردا فردا من الاستقبال ثم الطاقم الطبي، محترمين وبشوشين"
    predictions = predict_feedback(sentence)
    labels = ["Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"]

    for label, prob in zip(labels, predictions[0]):
        print(f"{label}: {prob:.4f}")