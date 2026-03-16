"""
rStar: MCTS-guided reasoning trace generation using Phi-4-mini.

Each node in the tree is a partial reasoning chain. At each step, the model
generates candidate next steps (children). We simulate to completion, score
with answer correctness, and backpropagate. After the search, we collect the
best reasoning trace.
"""

import math
import random
from dataclasses import dataclass, field
from mlx_lm import load, generate

MODEL = "mlx-community/Phi-4-mini-instruct-4bit"
model, tokenizer = load(MODEL)

C_EXPLORE = 1.4   # UCB exploration constant
N_SIMULATIONS = 20
N_CHILDREN = 3    # branching factor per expansion
MAX_STEPS = 5     # max reasoning steps per trace
MAX_TOKENS = 128


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    steps: list[str]          # reasoning steps accumulated so far
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def ucb(self, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.avg_reward
        explore = C_EXPLORE * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def chain(self) -> str:
        return "\n".join(self.steps)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _generate(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)


def generate_next_step(question: str, steps_so_far: list[str]) -> str:
    """Ask the model for the next single reasoning step."""
    chain = "\n".join(steps_so_far) if steps_so_far else "(none yet)"
    prompt = (
        f"Question: {question}\n\n"
        f"Reasoning so far:\n{chain}\n\n"
        "Write the NEXT single reasoning step only. Be concise."
    )
    return _generate(prompt).strip()


def extract_final_answer(question: str, steps: list[str]) -> str:
    """Given a complete reasoning chain, extract the final answer."""
    chain = "\n".join(steps)
    prompt = (
        f"Question: {question}\n\n"
        f"Reasoning:\n{chain}\n\n"
        "State only the final numeric answer, nothing else."
    )
    return _generate(prompt, max_tokens=32).strip()


def score(predicted: str, correct: str) -> float:
    """Binary reward: 1.0 if answers match, else 0.0."""
    pred = predicted.strip().lower().replace(" ", "")
    truth = correct.strip().lower().replace(" ", "")
    return 1.0 if pred == truth else 0.0


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

def select(node: Node) -> Node:
    """Traverse tree following max UCB until we reach a leaf."""
    while not node.is_leaf():
        node = max(node.children, key=lambda c: c.ucb(node.visits))
    return node


def expand(node: Node, question: str) -> None:
    """Generate N_CHILDREN candidate next steps and attach as children."""
    for _ in range(N_CHILDREN):
        next_step = generate_next_step(question, node.steps)
        child = Node(steps=node.steps + [next_step], parent=node)
        node.children.append(child)


def simulate(node: Node, question: str, correct_answer: str) -> float:
    """Roll out from this node to a final answer and return the reward."""
    steps = list(node.steps)

    # continue generating steps until MAX_STEPS
    while len(steps) < MAX_STEPS:
        next_step = generate_next_step(question, steps)
        steps.append(next_step)
        # early stop if the step looks like a final answer
        if any(tok in next_step.lower() for tok in ["therefore", "answer is", "= ", "equals"]):
            break

    predicted = extract_final_answer(question, steps)
    return score(predicted, correct_answer)


def backpropagate(node: Node, reward: float) -> None:
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent


def best_trace(root: Node) -> list[str]:
    """Walk the tree greedily by avg_reward to recover the best trace."""
    node = root
    while not node.is_leaf():
        node = max(node.children, key=lambda c: c.avg_reward)
    return node.steps


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def rstar_search(question: str, correct_answer: str) -> dict:
    root = Node(steps=[])

    for i in range(N_SIMULATIONS):
        print(f"  simulation {i + 1}/{N_SIMULATIONS}...", end="\r")

        leaf = select(root)

        # expand if not too deep
        if len(leaf.steps) < MAX_STEPS:
            expand(leaf, question)
            leaf = random.choice(leaf.children)

        reward = simulate(leaf, question, correct_answer)
        backpropagate(leaf, reward)

    print()

    trace = best_trace(root)
    final_answer = extract_final_answer(question, trace)

    return {
        "question": question,
        "correct_answer": correct_answer,
        "reasoning_trace": trace,
        "predicted_answer": final_answer,
        "correct": score(final_answer, correct_answer) == 1.0,
        "root_visits": root.visits,
    }


# ---------------------------------------------------------------------------
# Sample run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple arithmetic sample — no file I/O needed to demonstrate
    sample_question = "What is 76 * 13?"
    sample_answer = "988"

    print("=" * 60)
    print(f"Question : {sample_question}")
    print(f"Answer   : {sample_answer}")
    print("=" * 60)
    print(f"Running MCTS ({N_SIMULATIONS} simulations, branching={N_CHILDREN})...\n")

    result = rstar_search(sample_question, sample_answer)

    print("\nBest reasoning trace:")
    for i, step in enumerate(result["reasoning_trace"], 1):
        print(f"  Step {i}: {step}")

    print(f"\nPredicted : {result['predicted_answer']}")
    print(f"Correct   : {result['correct_answer']}")
    print(f"Match     : {'YES' if result['correct'] else 'NO'}")
    print(f"Tree nodes visited (root): {result['root_visits']}")
