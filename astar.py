"""
A* pathfinding on a 2D grid.

A* finds the shortest path between two points by combining:
  - g(n): actual cost from start to node n
  - h(n): heuristic estimate from n to goal (Manhattan distance)
  - f(n) = g(n) + h(n): priority used to explore nodes

Nodes with the lowest f(n) are explored first via a min-heap.
"""

import heapq
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Point(NamedTuple):
    row: int
    col: int


WALL = "#"
OPEN = "."
START = "S"
GOAL = "G"
PATH = "*"

MOVES = [
    Point(-1,  0),  # up
    Point( 1,  0),  # down
    Point( 0, -1),  # left
    Point( 0,  1),  # right
]


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def manhattan(a: Point, b: Point) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

def astar(grid: list[list[str]], start: Point, goal: Point) -> list[Point] | None:
    """
    Returns the shortest path from start to goal as a list of Points,
    or None if no path exists.
    """
    rows, cols = len(grid), len(grid[0])

    # min-heap entries: (f, g, point)
    heap: list[tuple[int, int, Point]] = []
    heapq.heappush(heap, (0, 0, start))

    came_from: dict[Point, Point | None] = {start: None}
    g_score: dict[Point, int] = {start: 0}

    while heap:
        _, g, current = heapq.heappop(heap)

        if current == goal:
            return _reconstruct(came_from, current)

        # skip if we already found a better path to this node
        if g > g_score.get(current, float("inf")):
            continue

        for move in MOVES:
            neighbor = Point(current.row + move.row, current.col + move.col)

            if not (0 <= neighbor.row < rows and 0 <= neighbor.col < cols):
                continue
            if grid[neighbor.row][neighbor.col] == WALL:
                continue

            tentative_g = g + 1  # uniform step cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(heap, (f, tentative_g, neighbor))
                came_from[neighbor] = current

    return None  # no path found


def _reconstruct(came_from: dict[Point, Point | None], current: Point) -> list[Point]:
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def parse_grid(text: str) -> tuple[list[list[str]], Point, Point]:
    grid = [list(row) for row in text.strip().splitlines()]
    start = goal = None
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == START:
                start = Point(r, c)
            elif cell == GOAL:
                goal = Point(r, c)
    assert start and goal, "Grid must contain S (start) and G (goal)"
    return grid, start, goal


def render(grid: list[list[str]], path: list[Point] | None) -> str:
    display = [row[:] for row in grid]
    if path:
        for p in path:
            if display[p.row][p.col] not in (START, GOAL):
                display[p.row][p.col] = PATH
    return "\n".join("".join(row) for row in display)


# ---------------------------------------------------------------------------
# Sample run
# ---------------------------------------------------------------------------

SAMPLE_GRID = """
S . . . # . . .
. # # . # . # .
. # . . . . # .
. . . # # . . .
# # . . . # . .
. . . # . . . G
"""

if __name__ == "__main__":
    grid, start, goal = parse_grid(SAMPLE_GRID)

    print("Grid:")
    print(render(grid, None))
    print(f"\nStart: {start}  Goal: {goal}")
    print("\nSearching...\n")

    path = astar(grid, start, goal)

    if path is None:
        print("No path found.")
    else:
        print("Path found:")
        print(render(grid, path))
        print(f"\nPath length : {len(path) - 1} steps")
        print(f"Nodes in path: {path}")
