# ECHO — Speech You Can Feel

A real-time speech-to-haptic communication system for DeafBlind users. ECHO listens to spoken
language, understands its meaning (not just its words), and encodes that meaning into tactile
feedback the user can feel directly — no interpreter, no lag, no dependence on an internet
connection to work.

> ✏️ **TODO:** Add a one-line project tagline/hook here if you want something punchier than the
> title alone, and a link to your demo video once you have one.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [The Dataset](#the-dataset)
- [The Edge Model](#the-edge-model)
- [Fallback Options](#fallback-options)
- [Deploying to Snapdragon Hardware](#deploying-to-snapdragon-hardware)
- [Known Limitations & Status](#known-limitations--status)
- [Roadmap](#roadmap)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Pipeline:

```
Microphone → Whisper (speech-to-text) → text normalization → multi-head classifier → haptic encoding
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
   → On-device LLM via GenieX + Hexagon NPU (still fully offline)
      → Cloud LLM (rare, last resort — only if the first two both fail)
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
6. **Haptic encoding** — the final four-field output is mapped to a tactile 3×2 matrix pattern.

> ✏️ **TODO:** This last step (haptic encoding → physical output) isn't documented yet in this
> README — add a section here once the haptic hardware/encoding scheme is finalized (what pattern
> maps to what signal, what hardware you're driving, wiring diagram if relevant).

---

## Repository Structure

> ✏️ **TODO:** This layout reflects what's been built so far — reorganize into your actual repo
> folders (e.g. `dataset/`, `model/`, `fallback/`, `docs/`) and update paths below to match.

```
.
├── generator_source/
│   ├── lexicon.py              # vocab, grammar helpers, casual/ASR paraphrase generation
│   ├── generators.py           # per-action sentence template generators
│   ├── main.py                 # runs generators → dedupes → splits → writes main CSV
│   ├── challenge_gen.py        # builds the adversarial evaluation set
│   └── validate_and_report.py  # QC checks + dataset_report.txt
├── haptic_dataset_v4_100k.csv  # main train/validation/test set (see note below on filename)
├── haptic_challenge_test.csv   # adversarial "challenge" evaluation set
├── dataset_report.txt          # full distribution/QC report for the above
├── trainmodel_cnn.py           # multi-head Text-CNN training + ONNX export
├── cloud_fallback.py           # cloud fallback client (Qualcomm Cloud AI 100 / Cirrascale)
├── cloud_fallback_grok.py      # cloud fallback client (xAI Grok, alternative provider)
├── local_ollama_fallback.py    # local LLM fallback via Ollama (CPU-only on ARM64)
├── local_geniex_fallback.py    # local LLM fallback via GenieX (NPU-accelerated)
└── ECHO_Hackathon_Deck.pptx    # presentation deck
```

> ⚠️ **Naming note:** `haptic_dataset_v4_100k.csv` currently contains **~6,080 rows**, not
> 100k — the "100k" in the filename is a holdover from the original spec target and no longer
> matches reality (dataset size was deliberately kept smaller for quality; see `dataset_report.txt`
> for the reasoning). Consider renaming the file before submitting/publishing so it doesn't
> mislead anyone skimming the repo.

---

## Getting Started

> ✏️ **TODO:** Fill in real setup steps once you've finalized your environment. Draft below —
> replace/verify each command actually works on a clean checkout.

```bash
# 1. Clone the repo
git clone <TODO: your repo URL>
cd <TODO: repo name>

# 2. Install Python dependencies
pip install -r requirements.txt   # TODO: this file doesn't exist yet — generate one
                                   # (pandas, numpy, tensorflow, scikit-learn, tf2onnx, requests,
                                   #  python-dotenv are all used somewhere in this repo)

# 3. Generate the dataset (optional — pre-built CSVs are already committed)
cd generator_source
python3 main.py
python3 challenge_gen.py
python3 validate_and_report.py

# 4. Train the edge model
cd ..
python3 trainmodel_cnn.py
```

> ✏️ **TODO:** Add instructions for running the full live pipeline (microphone → Whisper →
> model → haptic output) once that integration script exists — this README currently only
> covers the dataset/model/fallback pieces built so far.

---

## The Dataset

Built from scratch rather than sourced, because no off-the-shelf dataset teaches a model to
say "I don't know" instead of hallucinating.

- **~6,080** training examples, generated from ~185 controlled sentence-template families ×
  multiple persons/concepts × casual/ASR-style paraphrase variants
- **~1,910** adversarial "challenge" test sentences targeting known failure modes specifically:
  concept hallucination, subject confusion, negation, unseen sentence structures, unknown words,
  ASR-style fragments, contractions, and ambiguous questions
- **0%** verbatim overlap between the challenge set and the training split
- Splits are assigned by **template family**, not by row, so paraphrases of one sentence
  structure can't leak across train/validation/test
- All of the spec's original hard-negative cases (e.g. `"I'm going to go."` → `move | none`,
  not a hallucinated destination) resolve correctly

Full distributions, QC checks, and worked examples are in `dataset_report.txt`.

> ✏️ **TODO:** If you expand the dataset further (more rows, more concepts) before the final
> submission, re-run `validate_and_report.py` and paste the updated top-line numbers here so
> this section doesn't go stale.

---

## The Edge Model

A multi-head **Text-CNN** (parallel `Conv1D` branches at kernel sizes 2/3/4 + global max-pooling,
feeding four independent softmax heads) — deliberately **not** an LSTM.

Why: Whisper hands the model a complete utterance at once, so there's no task reason to pay for
recurrence. LSTM's per-timestep dependency chain doesn't parallelize onto NPU matrix hardware the
way convolution does, and Qualcomm's QNN toolchain has first-class support for exactly this kind
of architecture. Exported to ONNX (`tf2onnx`) for deployment via Qualcomm's QNN runtime.

> ✏️ **TODO:** Once you have a full training run's real numbers, replace this paragraph with
> actual accuracy/loss per head, and a confusion-matrix or per-class breakdown if you have one:
>
> - Intent accuracy: `[ADD NUMBER]`
> - Subject accuracy: `[ADD NUMBER]`
> - Action accuracy: `[ADD NUMBER]`
> - Concept accuracy: `[ADD NUMBER]`
> - Training time / hardware used: `[ADD DETAILS]`

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

> ⚠️ Never commit a real `.env` file or paste real API keys into docs, commits, or issues.

---

## Deploying to Snapdragon Hardware

- **Laptop (Snapdragon X Elite, Windows ARM64):** install [GenieX](https://github.com/qualcomm/GenieX),
  then:
  ```
  geniex pull ai-hub-models/Qwen3-4B-Instruct-2507
  geniex serve
  ```
  This serves an OpenAI-compatible API at `http://127.0.0.1:18181/v1` that `local_geniex_fallback.py`
  talks to directly.
- **Phone (Snapdragon 8 Elite, Android):** GenieX also ships a Java/Kotlin SDK for in-app,
  in-process inference — no local HTTP server needed. See `qualcomm/ai-hub-apps` on GitHub for a
  working reference Android app.

> ✏️ **TODO:** This is the part of the project furthest from done — add real setup notes once
> you've actually gotten the Android integration running (what you had to change, what broke,
> what model you ended up using on-phone).

---

## Known Limitations & Status

Being upfront about what's solid vs. still rough, for judges and for future-you:

- **GenieX is developer preview** — expect rough edges; verify exact model slugs against the
  live [AI Hub catalog](https://aihub.qualcomm.com/models) since names can shift.
- **Edge model accuracy numbers are not yet finalized** — see the TODO in
  [The Edge Model](#the-edge-model).
- **Android/phone integration is not yet built** — current work covers dataset, edge model, and
  four fallback options; the mobile app itself is not in this repo yet.
- **`onnxruntime-qnn` vs. generic `onnxruntime`** — confirm the QNN execution provider is actually
  being picked up at inference time (`providers=["QNNExecutionProvider"]`), not silently falling
  back to CPU. This was an open issue during development.

> ✏️ **TODO:** Add/remove items here as things get resolved or as new rough edges show up —
> keep this section honest, it's more useful to reviewers than a README that claims everything works.

---

## Roadmap

- [ ] Port edge model + GenieX fallback to Android (Snapdragon 8 Elite SDK)
- [ ] Expand the fixed concept vocabulary based on real usage
- [ ] Pilot testing with DeafBlind community members
- [ ] Tune confidence thresholds against real (not just synthetic) usage data
- [ ] Finalize haptic hardware encoding scheme

> ✏️ **TODO:** Reorder/edit this list to match your actual post-hackathon plans, and check off
> anything you finish before submission.

---

## Team

> ✏️ **TODO:** Add team member names, roles, and contact info here.
>
> - `[Name]` — `[role, e.g. "model + dataset"]` — `[contact/GitHub]`
> - `[Name]` — `[role, e.g. "hardware + haptic encoding"]` — `[contact/GitHub]`
> - `[Name]` — `[role]` — `[contact/GitHub]`

---

## Acknowledgments

> ✏️ **TODO:** Credit the hackathon, Qualcomm/Snapdragon program contacts, and any mentors or
> resources that helped along the way. Example:
>
> Built at `[Hackathon Name]`, `[Date]`. Thanks to `[organizers/mentors]` and Qualcomm for
> Snapdragon hardware access and Cloud AI 100 credits.

---

## License

> ✏️ **TODO:** Pick a license (MIT is a common default for hackathon projects if you want it
> freely reusable) and add the corresponding `LICENSE` file to the repo root.