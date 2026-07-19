from pathlib import Path
import pickle
import re
import os
import sys
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from wifi_sender import send_csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from local_llm_fallback import LocalLlamaFallback

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
FALLBACK_THRESHOLD = float(os.getenv("HAPTIC_FALLBACK_THRESHOLD", "0.85"))

llm_fallback = None
llama_model_path = os.getenv("LLAMA_MODEL_PATH")
if llama_model_path:
    llm_fallback = LocalLlamaFallback(model_path=llama_model_path)
else:
    llm_fallback = LocalLlamaFallback(model_path=None)

# ==================================================
# LOAD TOKENIZER + LABEL ENCODERS
# ==================================================

print(f"Loading haptic model from: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


class SimpleTokenizer:
    def __init__(self, num_words=5000, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {oov_token: 1}
        self.index_word = {1: oov_token}

    def _tokenize(self, text):
        text = str(text).lower().strip()
        text = text.replace("'", " ")
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.split() if text else []

    def fit_on_texts(self, texts):
        counts = {}
        for text in texts:
            for token in self._tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        for idx, (token, _) in enumerate(ordered[: self.num_words - 1], start=2):
            self.word_index[token] = idx

        self.index_word = {idx: token for token, idx in self.word_index.items()}
        return self

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = self._tokenize(text)
            seq = [self.word_index.get(token, self.word_index[self.oov_token]) for token in tokens]
            sequences.append(seq)
        return sequences


def _clean_sentence(sentence):
    sentence = str(sentence).lower().strip()

    contractions = {
        "i'm": "i am",
        "im": "i am",
        "i've": "i have",
        "i'll": "i will",
        "i'd": "i would",
        "you're": "you are",
        "youre": "you are",
        "you've": "you have",
        "you'll": "you will",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "couldn't": "could not",
        "wouldn't": "would not",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "that's": "that is",
        "what's": "what is",
        "where's": "where is",
        "how's": "how is",
        "let's": "let us",
    }

    for contraction, replacement in contractions.items():
        sentence = sentence.replace(contraction, replacement)

    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post"):
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for idx, seq in enumerate(sequences):
        seq = list(seq[:maxlen])
        if padding == "post":
            padded[idx, : len(seq)] = np.array(seq, dtype=np.int32)
        else:
            padded[idx, -len(seq):] = np.array(seq, dtype=np.int32)
    return padded


def build_text_processor_if_missing():
    dataset_path = QUALCOMM_HACK_DIR / "haptic_dataset_v4_100knew.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path).dropna()
    df["input_text"] = df["input_text"].apply(_clean_sentence)

    tokenizer = SimpleTokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["input_text"])

    le_intent = LabelEncoder()
    le_subject = LabelEncoder()
    le_action = LabelEncoder()
    le_concept = LabelEncoder()

    le_intent.fit(df["intent"])
    le_subject.fit(df["subject"])
    le_action.fit(df["action"])
    le_concept.fit(df["concept"])

    with open(PROCESSOR_PATH, "wb") as f:
        pickle.dump(
            {
                "tokenizer": tokenizer,
                "le_intent": le_intent,
                "le_subject": le_subject,
                "le_action": le_action,
                "le_concept": le_concept,
            },
            f,
        )

    return tokenizer, le_intent, le_subject, le_action, le_concept


if not PROCESSOR_PATH.exists():
    print(f"Processor not found at {PROCESSOR_PATH}, building it from dataset...")
    tokenizer, le_intent, le_subject, le_action, le_concept = build_text_processor_if_missing()
else:
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
    return _clean_sentence(sentence)

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

def _should_fallback(confidence_dict):
    return (
        confidence_dict.get("intent", 0.0) < FALLBACK_THRESHOLD
        or confidence_dict.get("subject", 0.0) < FALLBACK_THRESHOLD
        or confidence_dict.get("action", 0.0) < FALLBACK_THRESHOLD
        or confidence_dict.get("concept", 0.0) < FALLBACK_THRESHOLD
    )


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

    if _should_fallback(result["confidence"]):
        print(f"[LLM FALLBACK] confidence below {FALLBACK_THRESHOLD}: sending prompt to local LLM")
        fallback_result = llm_fallback.predict(original_sentence)
        if fallback_result.get("success"):
            fallback_result["confidence"] = result["confidence"]
            fallback_result["unknown_words"] = result.get("unknown_words", [])
            fallback_result["sentence"] = sentence
            fallback_result["source"] = "llama_fallback"

            send_csv(
                fallback_result["intent"],
                fallback_result["subject"],
                fallback_result["action"],
                fallback_result["concept"],
            )

            return fallback_result
        print("[LLM FALLBACK] local LLM did not return a valid parse")

    send_csv(
        result["intent"],
        result["subject"],
        result["action"],
        result["concept"],
    )

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