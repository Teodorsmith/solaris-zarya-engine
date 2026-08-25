from pathlib import Path

import pytest

from agent.engine.state_machine import TaskFSM
from agent.models import TaskState


def test_atomic_write_and_idempotency(tmp_path: Path):
    fsm = TaskFSM(tmp_path / "active_task.json", tmp_path / "state_manifest.json")
    state = fsm.start_task("goal_123")
    assert state.state == "PENDING"

    # advance state
    fsm.advance("RUNNING", action_hash="hash_abc")

    loaded = fsm.load_state()
    assert loaded.state == "RUNNING"
    assert loaded.step_index == 1

    key = fsm.get_idempotency_key()
    assert key == "goal_123_1_hash_abc"

    # Test corruption recovery
    with open(tmp_path / "active_task.json", "w") as f:
        f.write("{ invalid json")

    assert fsm.load_state() is None


# ── Schema tests ──────────────────────────────────────────────────────────────


def test_taskstate_accepts_all_literals():
    """Every state the FSM writes must be accepted by the Pydantic schema."""
    all_states = [
        "PENDING",
        "RUNNING",
        "VERIFYING",
        "COMMITTED",
        "COMPLETED",
        "FAILED",
        "ABORTED",
        "CANCELLED",
    ]
    for s in all_states:
        ts = TaskState(state=s)
        assert ts.state == s, f"State '{s}' rejected by TaskState schema"


def test_taskstate_rejects_unknown_state():
    """Unknown state values must raise a ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TaskState(state="BOGUS")


def test_completed_state_roundtrip(tmp_path: Path):
    """COMPLETED must survive a write/read cycle without Pydantic error."""
    fsm = TaskFSM(tmp_path / "active_task.json", tmp_path / "state_manifest.json")
    fsm.start_task("goal_rt")
    fsm.advance("RUNNING")

    # Manually write COMPLETED to disk (simulates run_to_completion finishing)
    state = TaskState(goal_id="goal_rt", state="COMPLETED")
    (tmp_path / "active_task.json").write_text(
        state.model_dump_json(), encoding="utf-8"
    )

    loaded = fsm.load_state()
    assert loaded is not None
    assert loaded.state == "COMPLETED"


# ── Manifest lifecycle tests ──────────────────────────────────────────────────


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED", "ABORTED", "CANCELLED"])
def test_terminal_state_clears_active_task(tmp_path: Path, terminal: str):
    """Advancing to any terminal state must delete active_task.json so the
    next boot does not offer a stale resume prompt."""
    state_file = tmp_path / "active_task.json"
    manifest_file = tmp_path / "state_manifest.json"
    fsm = TaskFSM(state_file, manifest_file)

    fsm.start_task("goal_term")
    assert state_file.exists(), "active_task.json should exist after start_task"

    fsm.advance(terminal)

    assert not state_file.exists(), (
        f"active_task.json should be deleted after advancing to '{terminal}'"
    )


def test_non_terminal_state_preserves_active_task(tmp_path: Path):
    """Non-terminal advances must NOT delete active_task.json."""
    state_file = tmp_path / "active_task.json"
    fsm = TaskFSM(state_file, tmp_path / "state_manifest.json")
    fsm.start_task("goal_nt")

    for state in ("RUNNING", "VERIFYING", "COMMITTED"):
        # Restart so load_state always finds a file
        fsm.start_task("goal_nt")
        fsm.advance(state)
        assert state_file.exists(), (
            f"active_task.json must survive non-terminal state '{state}'"
        )


def test_corruption_auto_clears_manifest(tmp_path: Path):
    """A corrupt active_task.json must be cleared automatically so boot
    does not get stuck on a permanently unreadable state file."""
    state_file = tmp_path / "active_task.json"
    manifest_file = tmp_path / "state_manifest.json"
    fsm = TaskFSM(state_file, manifest_file)
    fsm.start_task("goal_corrupt")

    # Corrupt the file
    state_file.write_text("{ not valid json", encoding="utf-8")

    result = fsm.load_state()
    assert result is None
    assert not state_file.exists(), (
        "Corrupt state file should be cleared by load_state()"
    )
