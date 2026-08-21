"""Task Planner: Decomposes tasks into Goal DAGs."""
import logging
from agent.brains.base import BaseBrain
from agent.models import Goal
from agent.memory.goals import GoalMemory

logger = logging.getLogger(__name__)

class TaskPlanner:
    def __init__(self, brain: BaseBrain, goal_memory: GoalMemory):
        self.brain = brain
        self.goal_memory = goal_memory

    def plan_task(self, prompt: str, is_autonomous: bool = False) -> list[Goal]:
        max_depth = 2 if is_autonomous else 4
        
        system_prompt = f"""
You are a Task Planner. The user wants to: {prompt}

Decompose this into a Directed Acyclic Graph (DAG) of sub-goals.
Maximum depth allowed: {max_depth}.

For each goal, assign a required_tier:
- 0: Safe (Read-only, memory search)
- 1: Sandboxed (Synthesize/test code in sandbox)
- 2: Destructive/System (File writes, shell commands)

Output ONLY a JSON list of goals matching this schema:
[
  {{
    "id": "goal_1",
    "description": "Short description of the goal",
    "parent_id": null,
    "dependencies": [],
    "completion_criteria": "How to know this is done",
    "required_tier": 0
  }}
]
Use internal string IDs (like "goal_1") to set up dependencies.
"""
        response = self.brain.generate(system_prompt)
        parsed = self.brain.extract_json(response)
        
        if not parsed or not isinstance(parsed, list):
            raise ValueError(f"Failed to generate valid plan JSON. Output was: {response}")

        goals = []
        # Convert internal string IDs to actual UUIDs
        id_map = {}
        from uuid import uuid4
        
        for p in parsed:
            new_id = str(uuid4())
            id_map[p["id"]] = new_id
            
        for p in parsed:
            parent = id_map.get(p.get("parent_id")) if p.get("parent_id") else None
            deps = [id_map.get(d) for d in p.get("dependencies", []) if d in id_map]
            
            goal = Goal(
                id=id_map[p["id"]],
                description=p["description"],
                parent_id=parent,
                dependencies=deps,
                completion_criteria=p["completion_criteria"],
                required_tier=p["required_tier"],
                status="PENDING"
            )
            goals.append(goal)
            
        return goals

    def commit_plan(self, goals: list[Goal]) -> None:
        """Saves the approved plan to the goals database."""
        for g in goals:
            self.goal_memory.register(g)
