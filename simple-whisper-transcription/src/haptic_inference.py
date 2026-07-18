from pathlib import Path
import pickle
import re
import numpy as np
import onnxruntime as ort
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==================================================
# CONFIGURATION
# ==================================================

QUALCOMM_HACK_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = QUALCOMM_HACK_DIR / "haptic_model_cnn.onnx"
PROCESSOR_PATH = QUALCOMM_HACK_DIR / "text_processor.pkl"

MAX_SEQ_LENGTH = 15

MIN_INTENT_CONFIDENCE = 0.60
MIN_SUBJECT_CONFIDENCE = 0.60
MIN_ACTION_CONFIDENCE = 0.60
MIN_CONCEPT_CONFIDENCE = 0.60

# ==================================================
# LOAD TOKENIZER + LABEL ENCODERS
# ==================================================

print(f"Loading haptic model from: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not PROCESSOR_PATH.exists():
    raise FileNotFoundError(f"Processor not found: {PROCESSOR_PATH}")

with open(PROCESSOR_PATH, "rb") as f:
    processor = pickle.load(f)

tokenizer = processor["tokenizer"]
le_intent = processor["le_intent"]
le_subject = processor["le_subject"]
le_action = processor["le_action"]
le_concept = processor["le_concept"]

# ==================================================
# LOAD ONNX MODEL
# ==================================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# Map output names instead of assuming ONNX output order
output_names = [output.name for output in session.get_outputs()]

print("Haptic model loaded successfully.")
print("Model outputs:", output_names)

# ==================================================
# TEXT CLEANING
# ==================================================

def clean_sentence(sentence: str) -> str:
    sentence = sentence.lower().strip()

    contractions = {
        "i'm": "im",
        "i've": "ive",
        "i'll": "ill",
        "i'd": "id",
        "you're": "youre",
        "you've": "youve",
        "you'll": "youll",
        "don't": "dont",
        "doesn't": "doesnt",
        "didn't": "didnt",
        "can't": "cant",
        "couldn't": "couldnt",
        "wouldn't": "wouldnt",
        "won't": "wont",
        "isn't": "isnt",
        "aren't": "arent",
        "that's": "thats",
        "what's": "whats",
    }

    for contraction, replacement in contractions.items():
        sentence = sentence.replace(contraction, replacement)

    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()

def find_unknown_words(sentence: str):
    return [
        word for word in sentence.split()
        if word not in tokenizer.word_index
    ]

# ==================================================
# HAPTIC MODEL INFERENCE
# ==================================================
KNOWN_CONCEPTS = set(le_concept.classes_)

def handle_direct_concept(sentence):
    """Handle utterances consisting only of a known concept."""
    text = sentence.strip().replace(" ", "_")

    if text in KNOWN_CONCEPTS and text not in {"none", "unknown_concept"}:
        return {
            "success": True,
            "intent": "statement",
            "subject": "me",
            "action": "communicate",
            "concept": text,
            "confidence": {
                "intent": 1.0,
                "subject": 1.0,
                "action": 1.0,
                "concept": 1.0
            },
            "unknown_words": [],
            "sentence": sentence,
            "source": "direct_concept"
        }

    return None


def validate_prediction(sentence, prediction):
    """
    Prevent the neural model from inventing specific concepts
    that have no evidence in the sentence.
    """

    words = set(sentence.split())
    predicted_concept = prediction["concept"]

    # These concepts are abstract and do not need to literally
    # appear in the sentence.
    ABSTRACT_CONCEPTS = {
        "none",
        "unknown_concept",
        "destination"
    }

    if predicted_concept in ABSTRACT_CONCEPTS:
        return prediction

    # Convert concepts such as train_station -> train station
    concept_words = set(predicted_concept.replace("_", " ").split())

    # Check whether at least one concept word actually appears
    if not concept_words.intersection(words):
        prediction["original_concept"] = predicted_concept
        prediction["concept"] = "none"
        prediction["concept_corrected"] = True

    return prediction

def predict_haptic_context(sentence: str):

    original_sentence = sentence
    sentence = clean_sentence(sentence)
    direct_result = handle_direct_concept(sentence)

    if direct_result is not None:
        return direct_result
    if not sentence:
        return {
            "success": False,
            "reason": "empty_sentence",
            "sentence": original_sentence
        }

    unknown_words = find_unknown_words(sentence)

    sequence = tokenizer.texts_to_sequences([sentence])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_SEQ_LENGTH,
        padding="post",
        truncating="post"
    ).astype(np.int32)

    # Explicitly request outputs by their ONNX names
    outputs = session.run(None, {input_name: padded})

    # Build name -> output mapping
    output_map = {
        name: output
        for name, output in zip(output_names, outputs)
    }

    # tf2onnx should preserve these names
    try:
        intent_probs = output_map["intent_output"]
        subject_probs = output_map["subject_output"]
        action_probs = output_map["action_output"]
        concept_probs = output_map["concept_output"]

    except KeyError:
        # Some tf2onnx versions rename output tensors.
        # Match them by their number of classes instead.
        intent_probs = None
        subject_probs = None
        action_probs = None
        concept_probs = None

        for output in outputs:

            num_classes = output.shape[-1]

            if num_classes == len(le_intent.classes_):
                intent_probs = output

            elif num_classes == len(le_subject.classes_):
                subject_probs = output

            elif num_classes == len(le_action.classes_):
                action_probs = output

            elif num_classes == len(le_concept.classes_):
                concept_probs = output

        if any(x is None for x in [
            intent_probs,
            subject_probs,
            action_probs,
            concept_probs
        ]):
            raise RuntimeError(
                "Could not map ONNX outputs to the four model heads. "
                f"Outputs: {[(x.name, x.shape) for x in session.get_outputs()]}"
            )

    # ==================================================
    # DECODE PREDICTIONS
    # ==================================================

    intent_index = int(np.argmax(intent_probs[0]))
    subject_index = int(np.argmax(subject_probs[0]))
    action_index = int(np.argmax(action_probs[0]))
    concept_index = int(np.argmax(concept_probs[0]))

    intent = le_intent.inverse_transform([intent_index])[0]
    subject = le_subject.inverse_transform([subject_index])[0]
    action = le_action.inverse_transform([action_index])[0]
    concept = le_concept.inverse_transform([concept_index])[0]

    intent_confidence = float(intent_probs[0][intent_index])
    subject_confidence = float(subject_probs[0][subject_index])
    action_confidence = float(action_probs[0][action_index])
    concept_confidence = float(concept_probs[0][concept_index])

    confidence = {
        "intent": intent_confidence,
        "subject": subject_confidence,
        "action": action_confidence,
        "concept": concept_confidence
    }

    # ==================================================
    # CONFIDENCE FALLBACK
    # ==================================================

    low_confidence_outputs = []

    if intent_confidence < MIN_INTENT_CONFIDENCE:
        low_confidence_outputs.append("intent")

    if subject_confidence < MIN_SUBJECT_CONFIDENCE:
        low_confidence_outputs.append("subject")

    if action_confidence < MIN_ACTION_CONFIDENCE:
        low_confidence_outputs.append("action")

    if concept_confidence < MIN_CONCEPT_CONFIDENCE:
        low_confidence_outputs.append("concept")

    result = {
    "success": True,
    "intent": intent,
    "subject": subject,
    "action": action,
    "concept": concept,
    "confidence": {
        "intent": intent_confidence,
        "subject": subject_confidence,
        "action": action_confidence,
        "concept": concept_confidence
    },
    "unknown_words": unknown_words,
    "sentence": sentence,
    "source": "edge_model"
}

    result = validate_prediction(sentence, result)

    return result

    if low_confidence_outputs:
        result["reason"] = "low_confidence"
        result["low_confidence_outputs"] = low_confidence_outputs

    return result

# ==================================================
# DIRECT TEST
# ==================================================

if __name__ == "__main__":

    test_sentences = [
        "Do you need water?",
        "What are you doing?",
        "Watch out for the pole",
        "Where are you going?",
        "I am feeling tired",
        "Please repeat that",
        "Turn left"
    ]

    for sentence in test_sentences:

        print("\n" + "=" * 60)
        print(f"INPUT: {sentence}")

        result = predict_haptic_context(sentence)

        print("RESULT:", result)

        if result["success"]:
            print(
                f"HAPTIC OUTPUT: "
                f"{result['intent']} | "
                f"{result['subject']} | "
                f"{result['action']} | "
                f"{result['concept']}"
            )