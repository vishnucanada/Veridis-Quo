"""
Alpha-Beta Pruning on a Tic-Tac-Toe game tree.

Alpha-beta pruning is an optimization of minimax search. It cuts off branches
that cannot possibly affect the final decision, reducing nodes evaluated from
O(b^d) to O(b^(d/2)) in the best case.

  - Maximizer (X) tries to maximize the score
  - Minimizer (O) tries to minimize the score
  - Alpha: best score the maximizer is guaranteed so far
  - Beta:  best score the minimizer is guaranteed so far
  - Prune when alpha >= beta — the opponent will never allow this branch
"""

from __future__ import annotations

X = "X"
O = "O"
EMPTY = "."

Board = list[str]  # 9 cells, row-major


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def make_board() -> Board:
    return [EMPTY] * 9


def display(board: Board) -> str:
    rows = [" | ".join(board[i:i+3]) for i in range(0, 9, 3)]
    return "\n---------\n".join(rows)


def moves(board: Board) -> list[int]:
    return [i for i, c in enumerate(board) if c == EMPTY]


LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def winner(board: Board) -> str | None:
    for a, b, c in LINES:
        if board[a] == board[b] == board[c] != EMPTY:
            return board[a]
    return None


def terminal(board: Board) -> bool:
    return winner(board) is not None or not moves(board)


def evaluate(board: Board) -> int:
    w = winner(board)
    if w == X:
        return 1
    if w == O:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Alpha-Beta Pruning
# ---------------------------------------------------------------------------

nodes_evaluated = 0  # global counter to demonstrate pruning


def alphabeta(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
) -> int:
    global nodes_evaluated
    nodes_evaluated += 1

    if terminal(board):
        return evaluate(board)

    if is_maximizing:
        best = -float("inf")
        for move in moves(board):
            board[move] = X
            score = alphabeta(board, depth + 1, alpha, beta, False)
            board[move] = EMPTY

            best = max(best, score)
            alpha = max(alpha, best)

            if alpha >= beta:
                break  # beta cutoff — minimizer won't allow this branch

        return best

    else:
        best = float("inf")
        for move in moves(board):
            board[move] = O
            score = alphabeta(board, depth + 1, alpha, beta, True)
            board[move] = EMPTY

            best = min(best, score)
            beta = min(beta, best)

            if alpha >= beta:
                break  # alpha cutoff — maximizer won't allow this branch

        return best


def best_move(board: Board, is_maximizing: bool) -> int:
    """Return the index of the optimal move for the current player."""
    global nodes_evaluated
    nodes_evaluated = 0

    best_score = -float("inf") if is_maximizing else float("inf")
    chosen = -1
    player = X if is_maximizing else O

    for move in moves(board):
        board[move] = player
        score = alphabeta(board, 0, -float("inf"), float("inf"), not is_maximizing)
        board[move] = EMPTY

        if is_maximizing and score > best_score:
            best_score = score
            chosen = move
        elif not is_maximizing and score < best_score:
            best_score = score
            chosen = move

    return chosen


# ---------------------------------------------------------------------------
# Sample: X vs O, both playing optimally
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    board = make_board()

    # Pre-set a mid-game position to make it interesting
    # X has center, O has top-left
    board[4] = X
    board[0] = O

    print("Starting position:")
    print(display(board))
    print()

    turn = X  # X moves next (maximizing)
    move_num = 1

    while not terminal(board):
        is_max = (turn == X)
        move = best_move(board, is_max)
        board[move] = turn

        row, col = divmod(move, 3)
        print(f"Move {move_num} — {turn} plays ({row},{col}), nodes evaluated: {nodes_evaluated}")
        print(display(board))
        print()

        turn = O if turn == X else X
        move_num += 1

    w = winner(board)
    if w:
        print(f"Winner: {w}")
    else:
        print("Draw — both played optimally")
