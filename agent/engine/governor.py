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
        
        response = input("Approve? [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            self._log_decision("USER_APPROVED", f"User approved action: {action_description}")
            return True
        else:
            self._log_decision("USER_DENIED", f"User denied action (response='{response}'): {action_description}")
            return False

    def request_skill_write_permission(
        self,
        skill_name: str,
        file_path: str,
        code_preview: str,
        is_autonomous: bool = False,
    ) -> bool:
        """Enforce HITL approval before writing validated skill code to disk."""
        if is_autonomous:
            self._log_decision("DENIED", f"Autonomous skill write disabled without supervisor approval for: {skill_name} ({file_path})")
            return False

        clean_preview = code_preview[:120].replace("\n", " ")
        print(f"\n[GOVERNOR WAKE] Skill file write requires approval (Tier 2 Action):")
        print(f"Skill: {skill_name}")
        print(f"File : {file_path}")
        print(f"Preview: {clean_preview}...")

        response = input("Approve? [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            self._log_decision("USER_APPROVED", f"User approved skill file write: {skill_name} ({file_path})")
            return True
        else:
            self._log_decision("USER_DENIED", f"User denied skill file write (response='{response}'): {skill_name} ({file_path})")
            return False

    def request_file_write_permission(
        self,
        file_path: str,
        content_preview: str,
        goal_description: str | None = None,
        is_autonomous: bool = False,
    ) -> bool:
        """Enforce HITL approval before executing any file write mutation."""
        if is_autonomous:
            self._log_decision("DENIED", f"Autonomous file write denied for: {file_path}")
            return False

        clean_preview = content_preview[:120].replace("\n", " ")
        print(f"\n[GOVERNOR WAKE] File write requires approval (Tier 2 Action):")
        if goal_description:
            print(f"Goal: {goal_description}")
        print(f"File: {file_path}")
        print(f"Preview: {clean_preview}...")

        response = input("Approve? [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            self._log_decision("USER_APPROVED", f"User approved file write: {file_path}")
            return True
        else:
            self._log_decision("USER_DENIED", f"User denied file write (response='{response}'): {file_path}")
            return False

    def _log_decision(self, outcome: str, content: str) -> None:
        log = EpisodicLog(
            kind="system",
            content=f"[GOVERNOR {outcome}] {content}",
            outcome="success" if "APPROVED" in outcome else "failure"
        )
        self.episodic.log_event(log)
