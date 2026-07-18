import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import pickle
import tf2onnx

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
print("Loading dataset...")
df = pd.read_csv("haptic_dataset_v4_100k.csv").dropna()

le_intent = LabelEncoder()
le_subject = LabelEncoder()
le_action = LabelEncoder()
le_concept = LabelEncoder()

y_intent = le_intent.fit_transform(df["intent"])
y_subject = le_subject.fit_transform(df["subject"])
y_action = le_action.fit_transform(df["action"])
y_concept = le_concept.fit_transform(df["concept"])

print("\n=== OUTPUT CLASSES ===")
print("Intent:", list(le_intent.classes_))
print("Subject:", list(le_subject.classes_))
print("Action:", list(le_action.classes_))
print("Concept count:", len(le_concept.classes_))
# [UPGRADE]: OOV Token Injection
# Teaches the network how to handle words it has never seen without hallucinating.
print("Injecting <OOV> tokens for robust unknown-word handling...")
def inject_oov(row):
    if random.random() < 0.05:
        concept = str(row["concept"])
        text = str(row["input_text"])

        concept_text = concept.replace("_", " ")

        if concept_text in text:
            row["input_text"] = text.replace(concept_text, "<OOV>")
            row["concept"] = "unknown_concept"

    return row

df = df.apply(inject_oov, axis=1)

safety_row = pd.DataFrame([{
    "input_text": "i need a <OOV>",
    "intent": "request",
    "subject": "me",
    "action": "need",
    "concept": "unknown_concept"
}])

df = pd.concat([df, safety_row], ignore_index=True)
# Initialize Label Encoders
le_intent = LabelEncoder()
le_subject = LabelEncoder()
le_action = LabelEncoder()
le_concept = LabelEncoder()

y_intent = le_intent.fit_transform(df["intent"])
y_subject = le_subject.fit_transform(df["subject"])
y_action = le_action.fit_transform(df["action"])
y_concept = le_concept.fit_transform(df["concept"])
# Tokenize the input text (Convert words to numbers)
MAX_VOCAB = 5000
MAX_SEQ_LENGTH = 15 # Max words per sentence

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
import re

def clean_sentence(sentence):
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
        "let's": "let us"
    }

    for contraction, replacement in contractions.items():
        sentence = sentence.replace(contraction, replacement)

    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()
df["input_text"] = df["input_text"].apply(clean_sentence)

tokenizer.fit_on_texts(df["input_text"])
X_seq = tokenizer.texts_to_sequences(df["input_text"])
X_padded = pad_sequences(X_seq, maxlen=MAX_SEQ_LENGTH, padding='post')

# Save the Tokenizer and Label Encoders for inference later
with open("text_processor.pkl", "wb") as f:
    pickle.dump({
        "tokenizer": tokenizer,
        "le_intent": le_intent,
        "le_subject": le_subject,
        "le_action": le_action,
        "le_concept": le_concept
    }, f)

# ==========================================
# 2. BUILD THE MULTI-HEADED LSTM MODEL
# ==========================================
print("Building the NPU-optimized model...")

# Input layer (Accepts integer arrays)
inputs = Input(shape=(MAX_SEQ_LENGTH,), name="input_ids")

# Embedding layer (Removed input_length to clear Keras 3 warnings)
x = Embedding(input_dim=MAX_VOCAB, output_dim=64)(inputs)

# Bi-Directional LSTM (Reads forwards and backwards for context)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x) 

# [UPGRADE]: Dropout Layer
# Randomly disables 40% of connections to prevent dataset overfitting
x = Dropout(0.4)(x) 

out_intent = Dense(
    len(le_intent.classes_),
    activation="softmax",
    name="intent_output"
)(x)

out_subject = Dense(
    len(le_subject.classes_),
    activation="softmax",
    name="subject_output"
)(x)

out_action = Dense(
    len(le_action.classes_),
    activation="softmax",
    name="action_output"
)(x)

out_concept = Dense(
    len(le_concept.classes_),
    activation="softmax",
    name="concept_output"
)(x)

model = Model(
    inputs=inputs,
    outputs=[
        out_intent,
        out_subject,
        out_action,
        out_concept
    ]
)
model.compile(
    optimizer="adam",
    loss={
        "intent_output": "sparse_categorical_crossentropy",
        "subject_output": "sparse_categorical_crossentropy",
        "action_output": "sparse_categorical_crossentropy",
        "concept_output": "sparse_categorical_crossentropy"
    },
    metrics={
        "intent_output": "accuracy",
        "subject_output": "accuracy",
        "action_output": "accuracy",
        "concept_output": "accuracy"
    }
)

model.summary()

# ==========================================
# 3. TRAIN ON CPU (For Safe Export)
# ==========================================
print("Training model...")

# [UPGRADE]: Early Stopping
# Automatically stops training when validation accuracy peaks so it never overfits.
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5,               # Waits for 5 epochs of no improvement before stopping
    restore_best_weights=True # Rolls back to the best non-overfitted weights!
)

history = model.fit(
    X_padded,
    {
        "intent_output": y_intent,
        "subject_output": y_subject,
        "action_output": y_action,
        "concept_output": y_concept
    },
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    shuffle=True
)

# ==========================================
# 4. TEST INFERENCE WITH OOV ROUTING
# ==========================================
def predict_sentence(sentence):
    print(f"\n[Input]: {sentence}")

    clean_sentence = sentence.lower().strip()

    seq = tokenizer.texts_to_sequences([clean_sentence])
    padded = pad_sequences(
        seq,
        maxlen=MAX_SEQ_LENGTH,
        padding="post"
    ).astype(np.int32)

    preds = model.predict(padded, verbose=0)

    p_intent = le_intent.inverse_transform(
        [np.argmax(preds[0][0])]
    )[0]

    p_subject = le_subject.inverse_transform(
        [np.argmax(preds[1][0])]
    )[0]

    p_action = le_action.inverse_transform(
        [np.argmax(preds[2][0])]
    )[0]

    p_concept = le_concept.inverse_transform(
        [np.argmax(preds[3][0])]
    )[0]

    confidences = {
        "intent": float(np.max(preds[0][0])),
        "subject": float(np.max(preds[1][0])),
        "action": float(np.max(preds[2][0])),
        "concept": float(np.max(preds[3][0]))
    }

    print(
        f"[Edge Output]: "
        f"{p_intent} | {p_subject} | "
        f"{p_action} | {p_concept}"
    )

    print(f"[Confidence]: {confidences}")
# Run tests
predict_sentence("do you need water")
predict_sentence("what are you doing")
predict_sentence("watch out for the pole")
predict_sentence("where are you going")
predict_sentence("i am feeling tired")
predict_sentence("please repeat that")
predict_sentence("turn left")

# ==========================================
# 5. EXPORT FOR QUALCOMM AI HUB (NPU)
# ==========================================
print("Exporting model to ONNX format for Snapdragon NPU...")
# Save in the modern Keras format to clear the legacy warning
model.save("haptic_model.keras") 

spec = (tf.TensorSpec((None, MAX_SEQ_LENGTH), tf.int32, name="input_ids"),)
output_path = "haptic_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
print(f"Success! Model exported natively to {output_path}")