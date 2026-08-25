# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Pydantic schemas. No logic here — just shape."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Fact(BaseModel):
    id: int | None = None
    text: str
    confidence: float = 0.7
    source_type: Literal["seed", "learned", "user_corrected", "web_ingestion"] = "seed"
    topic: str | None = None
    created_at: str = Field(default_factory=_now)


class Passage(BaseModel):
    """Longer-form context than a Fact — a paragraph, not a single sentence."""

    id: int | None = None
    text: str
    topic: str | None = None
    source_type: Literal["seed", "learned", "web_ingestion"] = "seed"
    created_at: str = Field(default_factory=_now)


class EpisodicLog(BaseModel):
    id: int | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: Literal[
        "query",
        "answer",
        "refusal",
        "system",
        # Phase 4A events
        "stale_fact_alert",
        "self_model_update",
        "heartbeat_cycle",
        "security_violation",
    ] = "system"
    content: str
    outcome: Literal["success", "failure", "neutral"] = "neutral"
    prompt_hash: str | None = None
    strategy_label: str | None = None
    novelty_score: float | None = None
    reasoning_domain: str | None = None
    outcome_class: Literal["success", "failure", "divergent"] | None = None
    hypothesis_count: int = 1
    created_at: str = Field(default_factory=_now)


class Skill(BaseModel):
    """Phase 0: schema only. Nothing in Phase 0 writes one of these —
    it exists so Phase 2's synthesis pipeline has a stable table to build on."""

    id: int | None = None
    name: str
    description: str
    file_path: str | None = None
    verification_tier: Literal["mock", "real_local", "real_external"] = "mock"
    success_count: int = 0
    fail_count: int = 0
    created_at: str = Field(default_factory=_now)


class SkillResultSchema(BaseModel):
    """Immutable output schema enforced by the Phase 2 Validator.
    Any skill run must print a JSON object matching this schema."""

    skill_name: str
    status: Literal["ok", "error"]
    result: Any = None
    errors: list[str] = Field(default_factory=list)


class Project(BaseModel):
    id: int | None = None
    name: str
    root_path: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ProjectFile(BaseModel):
    id: int | None = None
    project_id: int
    path: str
    sha256_hash: str
    summary: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ProjectDecision(BaseModel):
    id: int | None = None
    project_id: int
    title: str
    content: str
    related_files_json: str = "[]"
    created_at: str = Field(default_factory=_now)


class Goal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    parent_id: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    status: Literal[
        "PENDING", "ACTIVE", "COMPLETED", "FAILED", "ABORTED", "CANCELLED"
    ] = "PENDING"
    completion_criteria: str
    required_tier: int = 0
    created_at: str = Field(default_factory=_now)


class TaskState(BaseModel):
    version: str = "1.0"
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    updated_at: str = Field(default_factory=_now)
    step_index: int = 0
    state: Literal[
        "PENDING",
        "RUNNING",
        "VERIFYING",
        "COMMITTED",
        "COMPLETED",
        "FAILED",
        "ABORTED",
        "CANCELLED",
    ] = "PENDING"
    consecutive_failures: int = 0
    goal_id: str | None = None
    action_hash: str | None = None
    pending_action_hash: str | None = None
    executed_actions: list[str] = Field(default_factory=list)
    prompt_hash: str | None = None
    strategy_label: str | None = None


# ---------------------------------------------------------------------------
# Phase 4B: Reasoning Substrate (Mitigations #61, #63, #65)
# ---------------------------------------------------------------------------


class SRTTrace(BaseModel):
    """Structured Reasoning Trace -- output format for reasoning benchmarks.

    A verified SRT means the conclusion *follows from the stated premises*
    under the named inference rule. It does NOT mean the premises are
    true in the real world. Explicitly: 'valid inference, not truth verification.'
    """

    conclusion: str
    premises: list[str]
    inference_rule: Literal[
        "transitive_implication",
        "modus_ponens",
        "modus_tollens",
        "disjunctive_syllogism",
        "de_morgan",
        "conjunction",
    ]
    rejected_hypotheses: list[dict[str, str]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ReasoningEpisode(BaseModel):
    """Full SHyAOEDRGL tuple for one reasoning event (Mitigation #61).

    Fields map to: State, Hypothesis, Action, Observation,
    Error, Diagnosis, Revised hypothesis, Generalized Lesson.
    """

    id: int | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str | None = None
    # SHyAOEDRGL fields
    state: str
    hypothesis: str
    action: str
    observation: str
    error: str | None = None
    diagnosis: str | None = None
    revised_hypo: str | None = None
    generalized_rule: str | None = None
    # Metadata
    strategy_label: str | None = None
    reasoning_domain: str | None = None
    outcome_class: Literal["success", "failure", "divergent"] = "success"
    hypothesis_count: int = Field(default=1, ge=1)
    verified: bool = False
    srt_json: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_now)
