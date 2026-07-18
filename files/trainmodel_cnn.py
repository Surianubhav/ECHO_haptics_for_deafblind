import os
import re
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Dropout,
    SpatialDropout1D,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
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
    "concept": "unknown_concept",
}])
df = pd.concat([df, safety_row], ignore_index=True)

# Re-fit encoders after the safety row / OOV relabeling (concept set is unchanged,
# but this keeps behavior identical to the original script)
le_intent = LabelEncoder()
le_subject = LabelEncoder()
le_action = LabelEncoder()
le_concept = LabelEncoder()

y_intent = le_intent.fit_transform(df["intent"])
y_subject = le_subject.fit_transform(df["subject"])
y_action = le_action.fit_transform(df["action"])
y_concept = le_concept.fit_transform(df["concept"])

# ==========================================
# TEXT CLEANING + TOKENIZATION (unchanged from the LSTM script)
# ==========================================
MAX_VOCAB = 5000
MAX_SEQ_LENGTH = 15  # Max words per sentence

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")


def clean_sentence(sentence):
    sentence = str(sentence).lower().strip()

    contractions = {
        "i'm": "i am", "im": "i am", "i've": "i have", "i'll": "i will",
        "i'd": "i would", "you're": "you are", "youre": "you are",
        "you've": "you have", "you'll": "you will", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "can't": "cannot",
        "couldn't": "could not", "wouldn't": "would not", "won't": "will not",
        "isn't": "is not", "aren't": "are not", "that's": "that is",
        "what's": "what is", "where's": "where is", "how's": "how is",
        "let's": "let us",
    }
    for contraction, replacement in contractions.items():
        sentence = sentence.replace(contraction, replacement)

    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


df["input_text"] = df["input_text"].apply(clean_sentence)

tokenizer.fit_on_texts(df["input_text"])
X_seq = tokenizer.texts_to_sequences(df["input_text"])
X_padded = pad_sequences(X_seq, maxlen=MAX_SEQ_LENGTH, padding="post")

with open("text_processor.pkl", "wb") as f:
    pickle.dump({
        "tokenizer": tokenizer,
        "le_intent": le_intent,
        "le_subject": le_subject,
        "le_action": le_action,
        "le_concept": le_concept,
    }, f)

# ==========================================
# 2. BUILD THE MULTI-HEADED TEXT-CNN MODEL
# ==========================================
# Why CNN instead of (Bi)LSTM here:
#   - Whisper already hands us the WHOLE utterance at once -- this is single-shot
#     sentence classification, not streaming/causal generation, so there's no task
#     reason to pay for recurrence.
#   - LSTM's per-timestep dependency chain is inherently sequential and doesn't
#     tile onto the Hexagon NPU's parallel matmul units the way conv/dense ops do;
#     QNN/Hexagon NN commonly falls back parts of an LSTM graph to CPU, which quietly
#     kills latency/battery on-device.
#   - Conv1D also quantizes to int8 more predictably than recurrent gating, which
#     matters for NPU deployment.
# Architecture: parallel Conv1D "n-gram detectors" (kernel sizes 2/3/4) + global
# max-pool, concatenated into a shared trunk, then 4 softmax heads. This is the
# classic Kim (2014) Text-CNN pattern -- cheap, fully parallel, well-supported by
# ONNX/QNN op sets.
print("Building the NPU-friendly Text-CNN model...")

EMBED_DIM = 64
FILTERS = 64
KERNEL_SIZES = (2, 3, 4)

inputs = Input(shape=(MAX_SEQ_LENGTH,), name="input_ids")

x = Embedding(input_dim=MAX_VOCAB, output_dim=EMBED_DIM)(inputs)
x = SpatialDropout1D(0.2)(x)  # regularize whole embedding channels, helps small datasets

conv_outputs = []
for k in KERNEL_SIZES:
    c = Conv1D(
        filters=FILTERS,
        kernel_size=k,
        activation="relu",
        padding="same",
        name=f"conv1d_k{k}",
    )(x)
    p = GlobalMaxPooling1D(name=f"maxpool_k{k}")(c)
    conv_outputs.append(p)

x = Concatenate(name="ngram_concat")(conv_outputs)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)

out_intent = Dense(len(le_intent.classes_), activation="softmax", name="intent_output")(x)
out_subject = Dense(len(le_subject.classes_), activation="softmax", name="subject_output")(x)
out_action = Dense(len(le_action.classes_), activation="softmax", name="action_output")(x)
out_concept = Dense(len(le_concept.classes_), activation="softmax", name="concept_output")(x)

model = Model(
    inputs=inputs,
    outputs=[out_intent, out_subject, out_action, out_concept],
)

model.compile(
    optimizer="adam",
    loss={
        "intent_output": "sparse_categorical_crossentropy",
        "subject_output": "sparse_categorical_crossentropy",
        "action_output": "sparse_categorical_crossentropy",
        "concept_output": "sparse_categorical_crossentropy",
    },
    metrics={
        "intent_output": "accuracy",
        "subject_output": "accuracy",
        "action_output": "accuracy",
        "concept_output": "accuracy",
    },
)

model.summary()

# ==========================================
# 3. TRAIN
# ==========================================
print("Training model...")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    X_padded,
    {
        "intent_output": y_intent,
        "subject_output": y_subject,
        "action_output": y_action,
        "concept_output": y_concept,
    },
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    shuffle=True,
)

# ==========================================
# 4. TEST INFERENCE
# ==========================================
def predict_sentence(sentence):
    print(f"\n[Input]: {sentence}")

    cleaned = clean_sentence(sentence)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding="post").astype(np.int32)

    preds = model.predict(padded, verbose=0)

    p_intent = le_intent.inverse_transform([np.argmax(preds[0][0])])[0]
    p_subject = le_subject.inverse_transform([np.argmax(preds[1][0])])[0]
    p_action = le_action.inverse_transform([np.argmax(preds[2][0])])[0]
    p_concept = le_concept.inverse_transform([np.argmax(preds[3][0])])[0]

    confidences = {
        "intent": float(np.max(preds[0][0])),
        "subject": float(np.max(preds[1][0])),
        "action": float(np.max(preds[2][0])),
        "concept": float(np.max(preds[3][0])),
    }

    print(f"[Edge Output]: {p_intent} | {p_subject} | {p_action} | {p_concept}")
    print(f"[Confidence]: {confidences}")
    print(f"[Min confidence]: {min(confidences.values()):.4f}")


predict_sentence("do you need water")
predict_sentence("what are you doing")
predict_sentence("watch out for the pole")
predict_sentence("where are you going")
predict_sentence("i am feeling tired")
predict_sentence("please repeat that")
predict_sentence("turn left")
predict_sentence("i am going to the workshop")     # regression check: should be move + unknown_concept
predict_sentence("i am going to try again")        # regression check: should be do + none
predict_sentence("i need a transistor")            # regression check: should be need + unknown_concept

# ==========================================
# 5. EXPORT FOR QUALCOMM AI HUB (NPU)
# ==========================================
print("Exporting model to ONNX format for Snapdragon NPU...")
model.save("haptic_model_cnn.keras")

spec = (tf.TensorSpec((None, MAX_SEQ_LENGTH), tf.int32, name="input_ids"),)
output_path = "haptic_model_cnn.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
print(f"Success! Model exported natively to {output_path}")
