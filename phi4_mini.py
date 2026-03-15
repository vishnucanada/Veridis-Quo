from mlx_lm import load, generate

# 4-bit quantized — fits comfortably in 16GB
MODEL = "mlx-community/Phi-4-mini-instruct-4bit"

model, tokenizer = load(MODEL)


def chat(prompt: str, max_tokens: int = 512, temp: float = 0.7) -> str:
    messages = [{"role": "user", "content": prompt}]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        temp=temp,
        verbose=False,
    )
    

    return response


if __name__ == "__main__":
    print("Phi-4 Mini")
    print("Type 'q' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "q":
            break
        if not user_input:
            continue

        response = chat(user_input)
        print(f"\nPhi-4: {response}\n")

