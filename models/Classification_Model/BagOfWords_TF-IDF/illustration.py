import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# Load the multi-output classifier:
model = joblib.load("saved_model_tfidf/multioutput_model.joblib")
vectorizer = joblib.load("saved_model_tfidf/vectorizer.joblib")
aspect_names = ["Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"]


def predict_review(review_text):
    X = vectorizer.transform([review_text])
    preds = model.predict(X)[0]
    # Convert numpy int64 to int
    preds = [int(p) for p in preds]
    return dict(zip(aspect_names, preds))

review = "الأسعار مرتفعة لكن الأطباء ممتازون"
prediction = predict_review(review)



print(review)
print(prediction)


# 🔹 Arabic review examples
reviews = [
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

# 🔹 Loop and print
for review in reviews:
    prediction = predict_review(review)
    print(f"Review: {review}")
    print(f"Prediction: {prediction}")
    print("-" * 60)