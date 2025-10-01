# test_inference.py
import builtins
from app.inference import InferenceEngine, OllamaClient, normalize_response, UNKNOWN_STRING

class DummyClient(OllamaClient):
    def __init__(self): pass
    def chat(self, **kwargs):
        # Simulate model uncertainty for obscure prompt
        return {"response": "I'm not sure about that.", "usage": {"tokens_prompt": 10, "tokens_output": 5, "total_tokens": 15}}

def test_normalize_uncertain_to_unknown():
    assert normalize_response("I am not sure.") == UNKNOWN_STRING
    assert normalize_response("") == UNKNOWN_STRING

def test_inference_unknown_policy():
    eng = InferenceEngine(DummyClient())
    resp, meta = eng.generate(prompt="gibberish qxzvwy??", language="en")
    assert resp == UNKNOWN_STRING
    assert "model" in meta and "usage" in meta
