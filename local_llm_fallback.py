import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests

ROOT_DIR = Path(__file__).resolve().parent


ALLOWED_INTENTS = {"alert", "question", "request", "statement"}
ALLOWED_SUBJECTS = {"me", "you", "other"}
ALLOWED_ACTIONS = {"communicate", "danger", "do", "feel", "move", "need"}


class LocalLlamaFallback:

    def __init__(self, model_path=None, base_url=None):
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH")
        self.base_url = base_url or os.getenv(
            "LLAMA_BASE_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        self.timeout = int(os.getenv("LLAMA_TIMEOUT", "60"))

        self.api_key = (
            os.getenv("LLAMA_API_KEY")
            or os.getenv("GROQ_API_KEY")
        )

        self.model_name = (
            os.getenv("LLAMA_MODEL_NAME")
            or os.getenv("GROQ_MODEL")
            or "llama-3.1-8b-instant"
        )

    ####################################################################
    # JSON PARSER
    ####################################################################

    def parse_response(self, text):

        if text is None:
            return None

        text = text.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        try:
            obj = json.loads(text)
        except Exception:
            print("[LLM] Could not decode JSON")
            print(text)
            return None

        intent = obj.get("intent", "").lower().strip()
        subject = obj.get("subject", "").lower().strip()
        action = obj.get("action", "").lower().strip()
        concept = obj.get("concept", "none").lower().strip()

        if intent not in ALLOWED_INTENTS:
            return None

        if subject not in ALLOWED_SUBJECTS:
            return None

        if action not in ALLOWED_ACTIONS:
            return None

        concept = re.sub(r"[^a-z0-9 '\-]", "", concept)
        concept = re.sub(r"\s+", "_", concept)

        if concept == "":
            concept = "none"

        return {
            "success": True,
            "source": "groq",
            "intent": intent,
            "subject": subject,
            "action": action,
            "concept": concept,
        }

    ####################################################################
    # PROMPT
    ####################################################################

    def _build_messages(self, sentence):

        system = """
You are a semantic parser.

Return ONLY valid JSON.

Never explain.

Never use markdown.

Never invent labels.

Allowed intents:

alert
question
request
statement

Allowed subjects:

me
you
other

Allowed actions:

communicate
danger
do
feel
move
need

Examples

Input:
What are you doing?

Output:

{
"intent":"question",
"subject":"you",
"action":"do",
"concept":"none"
}
Input:
What is your occupation?

Output:
{
"intent":"question",
"subject":"you",
"action":"communicate",
"concept":"occupation"
}

Input:
What is your name?

Output:
{
"intent":"question",
"subject":"you",
"action":"communicate",
"concept":"name"
}

Input:
Where do you work?

Output:
{
"intent":"question",
"subject":"you",
"action":"do",
"concept":"work"
}
Input:
Is your father a lawyer?

Output:

{
"intent":"question",
"subject":"other",
"action":"communicate",
"concept":"lawyer"
}

Input:
Give me the screwdriver.

Output:

{
"intent":"request",
"subject":"you",
"action":"need",
"concept":"screwdriver"
}
"""

        return [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": sentence,
            },
        ]

    ####################################################################
    # CLOUD
    ####################################################################

    def _run_openai_compatible(self, sentence):

        if not self.api_key:
            return None

        payload = {
            "model": self.model_name,
            "messages": self._build_messages(sentence),
            "temperature": 0,
            "max_tokens": 64,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        print("=" * 60)
        print("STATUS:", response.status_code)
        print("=" * 60)

        if response.status_code != 200:
            print(response.text)
            return None

        data = response.json()

        output = data["choices"][0]["message"]["content"]

        print("=" * 60)
        print("MODEL OUTPUT")
        print(output)
        print("=" * 60)

        return output

    ####################################################################
    # CLI
    ####################################################################

    def _run_cli(self, sentence):

        if not self.model_path:
            return None

        cli = shutil.which("llama-cli") or shutil.which("llama")

        if cli is None:
            return None

        messages = self._build_messages(sentence)

        prompt = (
            messages[0]["content"]
            + "\n\nSentence:\n"
            + sentence
        )

        cmd = [
            cli,
            "-m",
            self.model_path,
            "-p",
            prompt,
            "--temp",
            "0",
            "-n",
            "64",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return result.stdout.strip()

        except Exception:
            return None

    ####################################################################
    # MAIN
    ####################################################################

    def predict(self, sentence):

        if self.model_path:

            output = self._run_cli(sentence)

            parsed = self.parse_response(output)

            if parsed is not None:
                return parsed

        output = self._run_openai_compatible(sentence)

        parsed = self.parse_response(output)

        if parsed is not None:
            return parsed

        return {
            "success": False,
            "source": "groq",
            "reason": "llm_unavailable",
            "sentence": sentence,
        }