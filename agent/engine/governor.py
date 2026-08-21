"""Permission Governor: Enforces HITL Tiers and Depth Caps."""
import logging
from agent.memory.episodic import EpisodicMemory
from agent.models import EpisodicLog, Goal

logger = logging.getLogger(__name__)

class PermissionGovernor:
    def __init__(self, episodic: EpisodicMemory):
        self.episodic = episodic

    def calculate_depth(self, goal: Goal, all_goals: list[Goal]) -> int:
        depth = 0
        current = goal
        goal_map = {g.id: g for g in all_goals}
        while current.parent_id and current.parent_id in goal_map:
            depth += 1
            current = goal_map[current.parent_id]
        return depth

    def request_permission(self, action_description: str, goal: Goal, all_goals: list[Goal], is_autonomous: bool = False) -> bool:
        depth = self.calculate_depth(goal, all_goals)
        
        # Autonomous mode hard cap
        if is_autonomous and depth > 2:
            self._log_decision("DENIED", f"Autonomous mode depth cap exceeded (depth {depth} > 2) for goal: {goal.description}")
            return False

        # Supervised mode depth cap
        if not is_autonomous and depth > 4:
            self._log_decision("DENIED", f"Supervised mode depth cap exceeded (depth {depth} > 4) for goal: {goal.description}")
            return False

        requires_approval = False
        reason = ""

        if goal.required_tier >= 2:
            requires_approval = True
            reason = "Tier 2 Destructive Action"
        elif not is_autonomous and depth > 2:
            requires_approval = True
            reason = f"Depth {depth} requires explicit approval"

        if not requires_approval:
            self._log_decision("AUTO_APPROVED", f"Action safe: {action_description}")
            return True

        print(f"\n[GOVERNOR WAKE] Action requires approval ({reason}):")
        print(f"Goal: {goal.description}")
        print(f"Action: {action_description}")
        
        while True:
            response = input("Approve? [Y/n]: ").strip().lower()
            if response == 'y' or response == 'yes':
                self._log_decision("USER_APPROVED", f"User approved action: {action_description}")
                return True
            elif response == 'n' or response == 'no':
                self._log_decision("USER_DENIED", f"User denied action: {action_description}")
                return False

    def _log_decision(self, outcome: str, content: str) -> None:
        log = EpisodicLog(
            kind="system",
            content=f"[GOVERNOR {outcome}] {content}",
            outcome="success" if "APPROVED" in outcome else "failure"
        )
        self.episodic.log_event(log)
