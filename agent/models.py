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
from typing import Literal, Optional, Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Fact(BaseModel):
    id: Optional[int] = None
    text: str
    confidence: float = 0.7
    source_type: Literal["seed", "learned", "user_corrected"] = "seed"
    topic: Optional[str] = None
    created_at: str = Field(default_factory=_now)


class Passage(BaseModel):
    """Longer-form context than a Fact — a paragraph, not a single sentence."""
    id: Optional[int] = None
    text: str
    topic: Optional[str] = None
    source_type: Literal["seed", "learned"] = "seed"
    created_at: str = Field(default_factory=_now)


class EpisodicLog(BaseModel):
    id: Optional[int] = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: Literal["query", "answer", "refusal", "system"] = "query"
    content: str
    outcome: Literal["success", "failure", "neutral"] = "neutral"
    created_at: str = Field(default_factory=_now)


class Skill(BaseModel):
    """Phase 0: schema only. Nothing in Phase 0 writes one of these —
    it exists so Phase 2's synthesis pipeline has a stable table to build on."""
    id: Optional[int] = None
    name: str
    description: str
    file_path: Optional[str] = None
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
    id: Optional[int] = None
    name: str
    root_path: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ProjectFile(BaseModel):
    id: Optional[int] = None
    project_id: int
    path: str
    sha256_hash: str
    summary: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ProjectDecision(BaseModel):
    id: Optional[int] = None
    project_id: int
    title: str
    content: str
    related_files_json: str = "[]"
    created_at: str = Field(default_factory=_now)


class Goal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    parent_id: Optional[str] = None
    dependencies: list[str] = Field(default_factory=list)
    status: Literal["PENDING", "ACTIVE", "COMPLETED", "FAILED", "ABORTED", "CANCELLED"] = "PENDING"
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
    goal_id: Optional[str] = None
    action_hash: Optional[str] = None
    pending_action_hash: Optional[str] = None
    executed_actions: list[str] = Field(default_factory=list)
