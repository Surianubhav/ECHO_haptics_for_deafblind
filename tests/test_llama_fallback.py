from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_llm_fallback import LocalLlamaFallback


def test_parse_llm_output():
    parser = LocalLlamaFallback(model_path=None, base_url=None)
    result = parser.parse_response("request | me | need | water")

    assert result["success"] is True
    assert result["intent"] == "request"
    assert result["subject"] == "me"
    assert result["action"] == "need"
    assert result["concept"] == "water"
