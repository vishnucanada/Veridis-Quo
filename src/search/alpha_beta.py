"""
Alpha-Beta Pruning on a Tic-Tac-Toe game tree.

Alpha-beta pruning is an optimization of minimax search. It cuts off branches
that cannot possibly affect the final decision, reducing nodes evaluated from
O(b^d) to O(b^(d/2)) in the best case.

Classes:
  - Board: holds game state, renders, and checks win/draw conditions
  - AlphaBeta: search engine that operates on a Board
  - Game: orchestrates players and the game loop
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

class Board:
    X = "X"
    O = "O"
    EMPTY = "."

    LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
        (0, 4, 8), (2, 4, 6),             # diagonals
    ]

    def __init__(self, cells: list[str] | None = None):
        self.cells = cells[:] if cells else [self.EMPTY] * 9

    def copy(self) -> Board:
        return Board(self.cells)

    def place(self, index: int, player: str) -> None:
        self.cells[index] = player

    def undo(self, index: int) -> None:
        self.cells[index] = self.EMPTY

    def available_moves(self) -> list[int]:
        return [i for i, c in enumerate(self.cells) if c == self.EMPTY]

    def winner(self) -> str | None:
        for a, b, c in self.LINES:
            if self.cells[a] == self.cells[b] == self.cells[c] != self.EMPTY:
                return self.cells[a]
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or not self.available_moves()

    def evaluate(self) -> int:
        w = self.winner()
        if w == self.X:
            return 1
        if w == self.O:
            return -1
        return 0

    def __str__(self) -> str:
        rows = [" | ".join(self.cells[i:i+3]) for i in range(0, 9, 3)]
        return "\n---------\n".join(rows)


# ---------------------------------------------------------------------------
# AlphaBeta search engine
# ---------------------------------------------------------------------------

class AlphaBeta:
    def __init__(self):
        self.nodes_evaluated = 0

    def search(self, board: Board, is_maximizing: bool) -> int:
        self.nodes_evaluated = 0
        best_score = -float("inf") if is_maximizing else float("inf")
        best_move = -1
        player = Board.X if is_maximizing else Board.O

        for move in board.available_moves():
            board.place(move, player)
            score = self._alphabeta(board, -float("inf"), float("inf"), not is_maximizing)
            board.undo(move)

            if is_maximizing and score > best_score:
                best_score, best_move = score, move
            elif not is_maximizing and score < best_score:
                best_score, best_move = score, move

        return best_move

    def _alphabeta(self, board: Board, alpha: float, beta: float, is_maximizing: bool) -> int:
        self.nodes_evaluated += 1

        if board.is_terminal():
            return board.evaluate()

        if is_maximizing:
            best = -float("inf")
            for move in board.available_moves():
                board.place(move, Board.X)
                best = max(best, self._alphabeta(board, alpha, beta, False))
                board.undo(move)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break  # beta cutoff
            return best
        else:
            best = float("inf")
            for move in board.available_moves():
                board.place(move, Board.O)
                best = min(best, self._alphabeta(board, alpha, beta, True))
                board.undo(move)
                beta = min(beta, best)
                if alpha >= beta:
                    break  # alpha cutoff
            return best


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class Game:
    def __init__(self, board: Board | None = None):
        self.board = board or Board()
        self.engine = AlphaBeta()
        self.turn = Board.X
        self.move_num = 1

    def play(self) -> None:
        print("Starting position:")
        print(self.board)
        print()

        while not self.board.is_terminal():
            is_max = (self.turn == Board.X)
            move = self.engine.search(self.board, is_max)
            self.board.place(move, self.turn)

            row, col = divmod(move, 3)
            print(f"Move {self.move_num} — {self.turn} plays ({row},{col}), nodes evaluated: {self.engine.nodes_evaluated}")
            print(self.board)
            print()

            self.turn = Board.O if self.turn == Board.X else Board.X
            self.move_num += 1

        w = self.board.winner()
        print(f"Winner: {w}" if w else "Draw — both played optimally")
