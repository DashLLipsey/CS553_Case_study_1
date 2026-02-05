import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app  # assumes your main Gradio logic is in app.py

class Token:
    def __init__(self, token): self.token = token

def test_api_requires_token():
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "HF_TOKEN not set in environment"

    gen = app.respond(
        message="Hi",
        history=[],
        system_message="test system msg",
        max_tokens=8,
        temperature=0.2,
        top_p=0.9,
        hf_token=Token(hf_token),
        use_local_model=False,  # Test Hugging Face API
    )
    first = next(gen)
    assert "please log in" not in first.lower()  # shouldn't get warning
    assert isinstance(first, str)
    assert len(first) > 0

def test_local_model_works():
    # Test using distilgpt2 as local model
    gen = app.respond(
        message="Hi",
        history=[],
        system_message="You are a helpful assistant.",
        max_tokens=8,
        temperature=0.2,
        top_p=0.9,
        hf_token=Token(""),  # Not needed for local
        use_local_model=True,  # Use local pipeline
    )
    first = next(gen)
    assert isinstance(first, str)
    assert len(first) > 0
