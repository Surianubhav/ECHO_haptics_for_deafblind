import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

AI100_API_KEY = os.getenv("AI100_API_KEY")
AI100_API_URL = os.getenv("AI100_API_URL")
AI100_MODEL = os.getenv("AI100_MODEL")

ALLOWED_INTENTS = {"alert", "question", "request", "statement"}
ALLOWED_SUBJECTS = {"me", "you", "other"}
ALLOWED_ACTIONS = {"communicate", "danger", "do", "feel", "move", "need"}

SYSTEM_PROMPT = """
You are the cloud fallback semantic parser for ECHO, a real-time
speech-to-haptic communication system for DeafBlind users.

Speech is transcribed by Whisper and normally processed by a small edge
CNN classifier. You are called only when the edge classifier is uncertain.

Convert the spoken sentence into exactly four fields:

intent | subject | action | concept

INTENT must be exactly one of:
alert, question, request, statement

SUBJECT must be exactly one of:
me, you, other

ACTION must be exactly one of:
communicate, danger, do, feel, move, need

CONCEPT RULES:

The concept is the most important concrete object, place, direction,
feeling, attribute, or topic necessary to understand the sentence.

If no meaningful concept exists, use:
none

Never invent a concept.

Never replace an unknown object with a familiar but unrelated object.

Preserve unfamiliar concrete words exactly as they appear in the sentence.

Examples:

What are you doing?
question | you | do | none

How are you?
question | you | feel | none

What do you want?
question | you | need | none

I need water.
request | me | need | water

Your dad needs painkillers.
statement | other | need | painkillers

Watch out for the car!
alert | you | danger | car

Turn left.
request | you | move | left

What is your name?
question | you | communicate | name

I am tired.
statement | me | feel | tired

I am going to the hospital.
statement | me | move | hospital

I need spironolactone.
request | me | need | spironolactone

Give me the screwdriver.
request | you | need | screwdriver

OUTPUT RULES:

Return exactly one line:

intent | subject | action | concept

Do not output JSON.
Do not use Markdown.
Do not explain your reasoning.
Do not add any other text.
"""


def clean_concept(concept):
    concept = concept.lower().strip()

    # Keep letters, numbers, spaces, hyphens and apostrophes
    concept = re.sub(r"[^a-z0-9 '\-]", "", concept)

    # Convert spaces to underscores for internal representation
    concept = re.sub(r"\s+", "_", concept)

    return concept or "none"


def validate_cloud_output(text):
    # Sometimes models still add whitespace/newlines
    text = text.strip()

    # Use only the first non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return None

    output = lines[0]

    parts = [part.strip().lower() for part in output.split("|")]

    if len(parts) != 4:
        return None

    intent, subject, action, concept = parts

    if intent not in ALLOWED_INTENTS:
        return None

    if subject not in ALLOWED_SUBJECTS:
        return None

    if action not in ALLOWED_ACTIONS:
        return None

    concept = clean_concept(concept)

    return {
        "success": True,
        "source": "cloud",
        "intent": intent,
        "subject": subject,
        "action": action,
        "concept": concept
    }


def predict_via_cloud(sentence, debug=False):
    if not AI100_API_KEY:
        return {
            "success": False,
            "source": "cloud",
            "reason": "missing_api_key"
        }

    if not AI100_API_URL:
        return {
            "success": False,
            "source": "cloud",
            "reason": "missing_api_url"
        }

    headers = {
        "Authorization": f"Bearer {AI100_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": AI100_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": sentence
            }
        ],
        "temperature": 0,
        "max_tokens": 50
    }

    if debug:
        print(f"[DEBUG] POST {AI100_API_URL}")
        print(f"[DEBUG] model={AI100_MODEL}")

    import time
    start = time.time()

    try:
        response = requests.post(
            AI100_API_URL,
            headers=headers,
            json=payload,
            timeout=(10, 90)  # (connect_timeout, read_timeout) -- generous read timeout
                              # in case the inference hardware is cold-starting
        )

        if debug:
            print(f"[DEBUG] elapsed={time.time() - start:.1f}s")
            print(f"[DEBUG] status_code={response.status_code}")
            print(f"[DEBUG] body={response.text[:500]}")

        if response.status_code != 200:
            return {
                "success": False,
                "source": "cloud",
                "reason": "api_error",
                "status_code": response.status_code,
                "error": response.text[:500]
            }

        data = response.json()

        # OpenAI-compatible response format
        output = data["choices"][0]["message"]["content"]

        result = validate_cloud_output(output)

        if result is None:
            return {
                "success": False,
                "source": "cloud",
                "reason": "invalid_llm_output",
                "raw_output": output
            }

        result["sentence"] = sentence

        return result

    except requests.exceptions.ConnectTimeout:
        return {
            "success": False,
            "source": "cloud",
            "reason": "connect_timeout",
            "note": "Could not even open a connection -- likely a network/firewall/VPN "
                    "issue reaching this host, not a slow server."
        }

    except requests.exceptions.ReadTimeout:
        return {
            "success": False,
            "source": "cloud",
            "reason": "read_timeout",
            "note": "Connected fine, but the server never sent a response in time -- "
                    "likely a cold-starting model or an overloaded/hung backend."
        }

    except requests.Timeout:
        return {
            "success": False,
            "source": "cloud",
            "reason": "timeout"
        }

    except Exception as e:
        return {
            "success": False,
            "source": "cloud",
            "reason": "exception",
            "error": str(e)
        }


# ---------------------------------------------------------------------------
# IMPORTANT: this block must be at module level (column 0), NOT indented
# inside predict_via_cloud(). In the original file it was nested inside the
# function body, right after the final `except` clause -- since every branch
# of that function returns before reaching it, it was unreachable dead code
# and the test suite never actually ran.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ECHO Cloud Fallback Test")
    print("=" * 60)

    print(f"API key loaded: {bool(AI100_API_KEY)}")
    print(f"API URL: {AI100_API_URL}")
    print(f"Model: {AI100_MODEL}")

    tests = [
        "I need water",
        "Your dad needs spironolactone",
        "Please give me the screwdriver",
        "Watch out for the car",
        "What are you doing?",
        "How are you?",
        "I'm going to the hospital",
        "Give me the oscilloscope"
    ]

    for sentence in tests:
        print("\n" + "-" * 60)
        print(f"INPUT: {sentence}")

        result = predict_via_cloud(sentence, debug=True)

        print(f"RESULT: {result}")

        if result.get("success"):
            print(
                "[CLOUD HAPTIC OUTPUT] "
                f"{result['intent']} | "
                f"{result['subject']} | "
                f"{result['action']} | "
                f"{result['concept']}"
            )
        else:
            print(
                "[CLOUD ERROR] "
                f"{result.get('reason', 'unknown_error')}"
            )