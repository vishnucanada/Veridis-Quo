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

from src.model import chat

C_EXPLORE = 1.4
N_SIMULATIONS = 20
N_CHILDREN = 3
MAX_STEPS = 5
MAX_TOKENS = 128


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    steps: list[str]
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

def _generate_step(question: str, steps_so_far: list[str]) -> str:
    chain = "\n".join(steps_so_far) if steps_so_far else "(none yet)"
    prompt = (
        f"Question: {question}\n\n"
        f"Reasoning so far:\n{chain}\n\n"
        "Write the NEXT single reasoning step only. Be concise."
    )
    return chat(prompt, max_tokens=MAX_TOKENS).strip()


def _extract_answer(question: str, steps: list[str]) -> str:
    prompt = (
        f"Question: {question}\n\n"
        f"Reasoning:\n{chr(10).join(steps)}\n\n"
        "State only the final numeric answer, nothing else."
    )
    return chat(prompt, max_tokens=32).strip()


def _score(predicted: str, correct: str) -> float:
    pred = predicted.strip().lower().replace(" ", "")
    truth = correct.strip().lower().replace(" ", "")
    return 1.0 if pred == truth else 0.0


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(
        self,
        n_simulations: int = N_SIMULATIONS,
        n_children: int = N_CHILDREN,
        max_steps: int = MAX_STEPS,
    ):
        self.n_simulations = n_simulations
        self.n_children = n_children
        self.max_steps = max_steps

    def search(self, question: str, correct_answer: str) -> dict:
        root = Node(steps=[])

        for i in range(self.n_simulations):
            print(f"  simulation {i + 1}/{self.n_simulations}...", end="\r")

            leaf = self._select(root)

            if len(leaf.steps) < self.max_steps:
                self._expand(leaf, question)
                leaf = random.choice(leaf.children)

            reward = self._simulate(leaf, question, correct_answer)
            self._backpropagate(leaf, reward)

        print()

        trace = self._best_trace(root)
        final_answer = _extract_answer(question, trace)

        return {
            "question": question,
            "correct_answer": correct_answer,
            "reasoning_trace": trace,
            "predicted_answer": final_answer,
            "correct": _score(final_answer, correct_answer) == 1.0,
            "root_visits": root.visits,
        }

    def _select(self, node: Node) -> Node:
        while not node.is_leaf():
            node = max(node.children, key=lambda c: c.ucb(node.visits))
        return node

    def _expand(self, node: Node, question: str) -> None:
        for _ in range(self.n_children):
            next_step = _generate_step(question, node.steps)
            child = Node(steps=node.steps + [next_step], parent=node)
            node.children.append(child)

    def _simulate(self, node: Node, question: str, correct_answer: str) -> float:
        steps = list(node.steps)
        while len(steps) < self.max_steps:
            next_step = _generate_step(question, steps)
            steps.append(next_step)
            if any(tok in next_step.lower() for tok in ["therefore", "answer is", "= ", "equals"]):
                break
        predicted = _extract_answer(question, steps)
        return _score(predicted, correct_answer)

    def _backpropagate(self, node: Node, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _best_trace(self, root: Node) -> list[str]:
        node = root
        while not node.is_leaf():
            node = max(node.children, key=lambda c: c.avg_reward)
        return node.steps
