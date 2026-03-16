from mlx_lm import load, generate
from dotenv import load_dotenv
import os

load_dotenv()
DATA_PATH = os.environ.get("DATA_PATH")
print(DATA_PATH)
# Microsofts Model
MODEL = "mlx-community/Phi-4-mini-instruct-4bit"

model, tokenizer = load(MODEL)


def chat(prompt: str, max_tokens: int = 512) -> str:
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
        verbose=False,
    )


    return response


if __name__ == "__main__":
    print("Welcome to Veridis Quo\n")

    problems_path = os.path.join(DATA_PATH, "train-easy/arithmetic__div.txt")
    
    difficulty = os.path.join(DATA_PATH, "train-easy")
    print(os.listdir(difficulty))
    with open(problems_path, "r") as myfile:
        file_contents = myfile.readlines()
    
    question = file_contents[0]
    answer = file_contents[1]

    
    

    response = chat(question)
    print(f"{question}\n")
    print(f"\nPhi-4: {response}\n")
    print(f"\nActual Answer {answer}")

    re_prompt = chat(f"Given that the answer is {answer}, would you change your answer?")
    print(re_prompt)

