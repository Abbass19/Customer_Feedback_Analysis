import transformers
import sentence_transformers
import torch

print(transformers.__version__)
print(sentence_transformers.__version__)
print(torch.__version__)

"""
GLiNER_NER_Model.py
Show how to use the saved Solution 3 models for inference.
"""

import os
import joblib
import torch
from sentence_transformers import SentenceTransformer

# 1️⃣ Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2️⃣ Paths
save_dir = "./saved_model_sbert"  # adjust if needed
aspect_names = ["Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"]

# 3️⃣ Load the 5 classifiers
models = [joblib.load(os.path.join(save_dir, f"{aspect}_clf.joblib")) for aspect in aspect_names]

# 4️⃣ Load the sentence-transformer embedder
# Use a proper pretrained Arabic SBERT
embedder_path = os.path.join(save_dir, "sentence_transformer")
if os.path.exists(embedder_path):
    embedder = SentenceTransformer(embedder_path, device=device)
else:
    # fallback if not saved locally
    embedder = SentenceTransformer("asafaya/bert-base-arabic", device=device)

# 5️⃣ Prediction function
def predict_review(review_text):
    """
    Input: single Arabic review (string)
    Output: dictionary of aspect -> predicted label [0,1,2,3]
    """
    embedding = embedder.encode([review_text], convert_to_tensor=True)
    preds = [model.predict(embedding.cpu().numpy())[0] for model in models]
    return dict(zip(aspect_names, preds))

# 6️⃣ Test examples
examples = [
    "الأسعار مرتفعة جدًا لكن الأطباء ممتازون في التعامل مع المرضى.",
    "المواعيد دائماً متأخرة والخدمة في الاستقبال سيئة.",
    "الطاقم الطبي مهني للغاية ويشرح كل شيء بصبر ووضوح.",
    "المستشفى نظيف جدًا والخدمات الطارئة سريعة وفعالة.",
    "خدمة العملاء سيئة ولا يهتمون بملاحظات المرضى.",
    "الأطباء ممتازون لكن تكلفة العلاج غير مبررة.",
    "الاستقبال لطيف والمواعيد دقيقة لكن قسم الطوارئ مزدحم جدًا.",
    "المستشفى قديم والأجهزة الطبية تحتاج لتحديث عاجل.",
    "العناية بالمريض ممتازة والفريق الطبي متعاون دائمًا.",
    "الأسعار مرتفعة والمواعيد صعبة والحجز عبر الإنترنت غير عملي."
]

# 7️⃣ Run predictions and print
for review in examples:
    prediction = predict_review(review)
    print(f"Review: {review}")
    print(f"Prediction: {prediction}")
    print("-" * 60)
