# ECHO — Speech You Can Feel

A real-time speech-to-haptic communication system for Deaf-Blind users utiizing the Advance Braille Technique. ECHO listens to spoken
language, understands its meaning (not just its words), and encodes that meaning into tactile
feedback the user can feel directly — no interpreter, no lag, no dependence on an round the clock 
active internet connection to work.

---
## Video Demo

> For Visual Demonstrations and Proof of Concept  
> (Since actual Motor Vibrations can't be demonstrated Videographically)


https://github.com/user-attachments/assets/9cb3d185-2ff1-4900-bcb9-66afcc3b9361

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [The Edge Model](#the-edge-model)
- [Fallback Options](#fallback-options)
- [Known Limitations & Status](#known-limitations--status)
- [Roadmap](#roadmap)
- [Future Upgrades](#future-updates)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Pipeline:

```
Microphone → Whisper (speech-to-text) → text normalization → Edge Sentence classifier → Edge AI is confident about its inference -> haptic encoding
                                                                      ↓
                                                      Edge AI is unsure of its inference
                                                                      ↓
                                             AI sends collected data for cloud Inference by Llama AI Instance
                                                                      ↓
                                                Arduino Uno Q Edge Device receives Inference results
                                                                      ↓
                                                               haptic encoding
```

The classifier splits every sentence into four independent signals a haptic device can act on:

| Signal | Values |
|---|---|
| **Intent** | `alert`, `question`, `request`, `statement` |
| **Subject** | `me`, `you`, `other` |
| **Action** | `communicate`, `danger`, `do`, `feel`, `move`, `need` |
| **Concept** | a fixed vocabulary of everyday objects/places/feelings, plus `none` and `unknown_concept` |

Example: `"Turn left."` → `request | you | move | left`

When the edge model isn't confident, the system escalates through a cascade rather than guessing:

```
               Edge CNN (instant, on-device)
                            ↓
 On-device LLM via GenieX + Hexagon NPU (still fully offline)
                            ↓
Cloud LLM (rare, last resort — only if the first two both fail)
```

This matters specifically because it's an assistive device: a confident wrong answer is worse
than an honest "I'm not sure," and a communication aid someone depends on daily can't stop
working just because the Wi-Fi does.

---

## How It Works

1. **Whisper** transcribes speech to raw text.
2. **Text normalization** cleans casual/contraction-heavy speech (see `clean_sentence()` in
   `trainmodel_cnn.py`).
3. **The edge model** — a multi-head Text-CNN (not an LSTM; see [The Edge Model](#the-edge-model)
   for why) — predicts all four signals in one pass, each with its own confidence score.
4. **Confidence gating** — if `min(intent_conf, subject_conf, action_conf, concept_conf)` falls
   below a threshold, the sentence is escalated.
5. **Escalation cascade** — first to an on-device LLM (GenieX, running on the Hexagon NPU, still
   fully offline), and only as a last resort to a cloud LLM.
6. **Haptic encoding** — the final four-field output is mapped to a tactile 3×2 matrix pattern and 3 quick inference tactile.

---

## Hardware

<img width="1600" height="1200" alt="image" src="https://github.com/user-attachments/assets/b1aada9c-d25e-405c-9b01-492d49d255c7" /> 

<img width="1200" height="1600" alt="image" src="https://github.com/user-attachments/assets/c5826bac-e605-49fb-a384-4841dfd8e96f" />


---

## Repository Structure

```
.
├── arduino-q/                          # Arduino micro-controller integration
│   ├── python/
│   │   └── main.py
│   ├── sketch/
│   │   └── app.yaml
│   ├── .gitignore
│   └── README.md
│
├── simple-whisper-transcription/       # Audio capture & voice processing pipeline
│   ├── reference/
│   │   └── WhisperApp.py
│   ├── src/
│   │   ├── LiveTranscriber.py
│   │   ├── LiveTranscriber_standalone.py
│   │   ├── haptic_inference.py         # Formulates inference from text to haptic commands
│   │   ├── model.py
│   │   ├── requirements.txt
│   │   ├── standalone_model.py
│   │   ├── standalone_whisper.py
│   │   ├── test_mic.py
│   │   └── wifi_sender.py
│   ├── .gitignore
│   ├── BUILD_EXECUTABLE.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── README.md
│   ├── WhisperTranscriber.spec
│   ├── build-requirements.txt
│   ├── build.bat
│   ├── build.ps1
│   ├── build_executable.py
│   ├── diagnose_executable.bat
│   ├── extract_mel_filters.py
│   ├── mel_filters.npz
│   └── requirements.txt
│
├── generator_source/                   # Data Synthesizer (To Be Built / Moved)
│   ├── lexicon.py                      # Vocab, ASR paraphrase helpers
│   ├── generators.py                   # Per-action templates
│   ├── main.py                         # Compiles, dedupes, splits data
│   ├── challenge_gen.py                # Builds adversarial evaluation set
│   └── validate_and_report.py          # QC validation scripts
│
├── tests/                              # Unit & integration testing suite
│
├── .gitignore
├── README.md
│
# --- ML Models & Datasets ---
├── haptic_dataset_v4_100knew.csv      # Main dataset (active version from your explorer)
├── haptic_challenge_test.csv           # Adversarial evaluation dataset
├── dataset_report.txt                  # QC distribution metrics
├── haptic_model_cnn.onnx               # Compiled deployment model
├── trainmodel_cnn.py                   # Model training script
├── pipeline.py                         # End-to-end local routing system
│
# --- Fallback Clients & LLM Routers ---
├── cloud_fallback.py                   # Qualcomm Cloud AI 100 / Cirrascale router
├── cloud_fallback_grok.py              # Alternative xAI Grok provider
├── local_llm_fallback.py               # CPU-based fallback (via Ollama on ARM64)
└── localgeniefallback.py               # NPU-accelerated fallback (via GenieX)
```

---

## Getting Started

> Requires Python v3.11+, Arduino App Lab
  
```bash
# 1. Clone the repo
git clone https://github.com/Surianubhav/ECHO_haptics_for_deafblind.git
cd ECHO_haptics_for_deafblind

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Generate the dataset (optional — pre-built CSVs are already committed)
cd generator_source
python3 main.py
python3 challenge_gen.py
python3 validate_and_report.py

# 4. Train the edge model
cd ..
python3 trainmodel_cnn.py

# Arduino Uno Q setup
Run Applab and run cd arduino-q/
```

---

## The Edge Model

A multi-head **Text-CNN** (parallel `Conv1D` branches at kernel sizes 2/3/4 + global max-pooling,
feeding four independent softmax heads) — deliberately **not** an LSTM.

Why: Whisper hands the model a complete utterance at once, so there's no task reason to pay for
recurrence. LSTM's per-timestep dependency chain doesn't parallelize onto NPU matrix hardware the
way convolution does, and Qualcomm's QNN toolchain has first-class support for exactly this kind
of architecture. Exported to ONNX (`tf2onnx`) for deployment via Qualcomm's QNN runtime.

> Achieved Action Accuracy: `96%`

---

## Fallback Options

Four interchangeable fallback clients are included, all speaking the same
`intent | subject | action | concept` output format behind one OpenAI-compatible request shape —
swapping between them is a config change, not a rewrite:

| File | Backend | Runs where | Notes |
|---|---|---|---|
| `cloud_fallback.py` | Qualcomm Cloud AI 100 (Cirrascale) | Cloud | Hackathon-provided; subject to shared-capacity limits |
| `cloud_fallback_grok.py` | xAI Grok | Cloud | Paid, but fast and reliable infra |
| `local_ollama_fallback.py` | Ollama | Local (CPU only on ARM64) | No NPU acceleration currently |
| `local_geniex_fallback.py` | GenieX + Qualcomm AI Hub | Local (NPU-accelerated) | **Recommended** — fully offline, uses the Hexagon NPU |

> ✏️ **TODO:** Decide which of these ships in your final demo and delete/archive the others (or
> keep all four and just say so here) — right now the README presents them as equally live
> options, which may not match what you actually run on stage.

Required environment variables (`.env`), depending on which client you use:

```
# Cirrascale / Qualcomm Cloud AI 100
AI100_API_KEY=<TODO: your key — rotate if it's ever been pasted anywhere>
AI100_API_URL=https://aisuite.cirrascale.com/apis/v2/chat/completions
AI100_MODEL=Llama-3.1-8B

# xAI Grok (alternative)
GROK_API_KEY=<TODO: your key>

# Ollama / GenieX — no API key needed, both default to localhost
```

---

## Known Limitations & Status

- Unavailability of Real User dataset to train on device AI for personalisation.
- Availability of Internet Connectivity for fallback Support.
- Android/phone integration is not yet implemented.

---

## Roadmap

- [ ] Develop Edge Model + Llama Fallback (Cloud)
- [ ] Develop Low Latency Framework
- [ ] Expand the fixed concept vocabulary based on real usage
- [ ] Tune confidence thresholds against real usage data
- [ ] Finalize haptic hardware and development

---

## Future

- Integration with web applications to enable Video Transcription via Haptics and Visual 

---

## Team: ECHO

- Abhinav Saini  
- Anubhav Suri  
- Arsh Handa  
- Yash Pratap Singh  

---

## Acknowledgments

+ Qualcomm Multiverse Hackathon Organisation Team  
+ Whister + AI Hub  
+ Arduino Uno Q References and Docs  
+ Llama LLM  
+ Thanks to Qualcomm for Snapdragon X Elite Laptop.  

---

## License

This project is released under an <u><b>Open Source</u></b> license.
