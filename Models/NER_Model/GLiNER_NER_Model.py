import warnings
import transformers
from gliner import GLiNER
import os

# ----------------------------
# Suppress warnings
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

# ----------------------------
# Global variables (lazy-loaded)
# ----------------------------
model = None
DEFAULT_ENTITY_LABELS = [
    "اسم_الطبيب", "الدور_الوظيفي", "اسم_المستشفى", "القسم",
    "التخصص", "منطقة_الخدمة", "السعر", "الوقت",
    "الموقع", "جودة_الخدمة", "نوع_المشكلة", "نوع_العلاج"
]

# ----------------------------
# Function: extract entities as array
# ----------------------------
def extract_entities_array(text, labels=None, threshold=0.75):
    """
    Extract named entities from a single text.
    Returns a list of 12 elements (same order as DEFAULT_ENTITY_LABELS).
    Missing entities are returned as None.
    Lazy-loads the GLiNER model on first call.
    """
    global model

    # Lazy-load model
    if model is None:
        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

    if labels is None:
        labels = DEFAULT_ENTITY_LABELS

    # Predict entities
    entities_raw = model.predict_entities(text, labels=labels, threshold=threshold)
    entities_dict = {}
    for e in entities_raw:
        # Only keep the first occurrence per label
        if e["label"] not in entities_dict:
            entities_dict[e["label"]] = e["text"]

    # Build output list in the same order as labels
    output = [entities_dict.get(label, None) for label in labels]
    return output

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    text = "المستشفى ممتاز، الدكتور أحمد ممتاز"
    result = extract_entities_array(text)
    print(result)
