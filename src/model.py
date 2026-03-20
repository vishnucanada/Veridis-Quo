"""
Model interface for Phi-4-mini via mlx_lm.
Provides a single chat() entry point used across the project.
"""

import os
from mlx_lm import load, generate
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.environ.get("DATA_PATH")
MODEL_ID = "mlx-community/Phi-4-mini-instruct-4bit"

model, tokenizer = load(MODEL_ID)


def chat(prompt: str, max_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
