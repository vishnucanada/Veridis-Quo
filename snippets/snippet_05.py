from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import heapq

class Priority(Enum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1

@dataclass(order=True)
class Task:
    priority: int
    name: str = field(compare=False)
    description: str = field(compare=False, default="")
    dependencies: List[str] = field(compare=False, default_factory=list)
    completed: bool = field(compare=False, default=False)

class TaskScheduler:
    def __init__(self):
        self._heap: List[Task] = []
        self._lookup: Dict[str, Task] = {}

    def add_task(self, name: str, priority: Priority, description: str = "", deps: List[str] = None):
        task = Task(priority=priority.value, name=name, description=description,
                    dependencies=deps or [])
        heapq.heappush(self._heap, task)
        self._lookup[name] = task

    def next_task(self) -> Optional[Task]:
        while self._heap:
            task = heapq.heappop(self._heap)
            if not task.completed and self._deps_met(task):
                return task
        return None

    def complete(self, name: str):
        if name in self._lookup:
            self._lookup[name].completed = True

    def _deps_met(self, task: Task) -> bool:
        return all(self._lookup.get(d, Task(0, d, completed=True)).completed
                   for d in task.dependencies)

    def pending_count(self) -> int:
        return sum(1 for t in self._lookup.values() if not t.completed)

    def all_completed(self) -> bool:
        return all(t.completed for t in self._lookup.values())

def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    queue = [n for n, d in in_degree.items() if d == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result

if __name__ == "__main__":
    scheduler = TaskScheduler()
    scheduler.add_task("setup", Priority.HIGH, "Initialize environment")
    scheduler.add_task("build", Priority.HIGH, "Compile project", deps=["setup"])
    scheduler.add_task("test", Priority.MEDIUM, "Run test suite", deps=["build"])
    scheduler.add_task("deploy", Priority.LOW, "Deploy to production", deps=["test"])

    while not scheduler.all_completed():
        task = scheduler.next_task()
        if task:
            print(f"Running: {task.name} [{task.description}]")
            scheduler.complete(task.name)
        else:
            break
