"""
Veridis Quo — entry point.

Loads a question from the dataset, runs rStar MCTS reasoning,
and prints the best trace alongside the correct answer.
"""

import os
from src.model import DATA_PATH
from src.search.rstar import MCTS


def load_question(difficulty: str, topic: str) -> tuple[str, str]:
    path = os.path.join(DATA_PATH, difficulty, topic)
    with open(path, "r") as f:
        lines = f.readlines()
    return lines[0].strip(), lines[1].strip()


if __name__ == "__main__":
    difficulty = "train-easy"
    topic = "arithmetic__add_or_sub.txt"

    question, answer = load_question(difficulty, topic)

    print("=" * 60)
    print(f"Question : {question}")
    print(f"Answer   : {answer}")
    print("=" * 60)

    mcts = MCTS()
    result = mcts.search(question, answer)

    print("\nBest reasoning trace:")
    for i, step in enumerate(result["reasoning_trace"], 1):
        print(f"  Step {i}: {step}")

    print(f"\nPredicted : {result['predicted_answer']}")
    print(f"Correct   : {result['correct_answer']}")
    print(f"Match     : {'YES' if result['correct'] else 'NO'}")
    print(f"Tree nodes visited (root): {result['root_visits']}")
