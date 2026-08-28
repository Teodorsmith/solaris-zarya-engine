# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Persistent Self-Model (data/self_model.json).

Enforces Mitigations #40 (empirical competence matrix) and #52 (write-
protection + anti-tamper integrity).

Write rules
-----------
Only four paths are allowed to mutate the model:

  * ``update_competence(topic, passed, source)``  -- benchmark pass/fail,
    subprocess exit codes.
  * ``record_user_correction(topic, field, old_val, new_val)``  -- explicit
    user corrections.
  * ``update_from_heartbeat(domain_deltas)``  -- weekly episodic SQL agg.
  * ``increment_boot_count()``  -- called once at startup.

Tamper-detection (3-state rollback on ``load()``)
--------------------------------------------------
Case A  -- first boot (no hash in manifest): write defaults, store hashes.
Case B  -- hash matches: proceed normally.
Case C  -- main hash mismatches:
    C1 -- backup hash matches self_model_bak_hash  -> restore backup.
    C2 -- backup hash also mismatches (or backup missing)  -> reset to defaults.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.memory.episodic import EpisodicMemory
    from agent.memory.state_manifest import StateManifest

logger = logging.getLogger(__name__)

_DEFAULT_MODEL: dict[str, Any] = {
    "identity": "Autonomous-Agent-v1",
    "boot_count": 0,
    "last_reflection_at": None,
    "current_focal_areas": [],
    "empirical_competence_matrix": {},
    "known_strengths": [],
    "known_knowledge_gaps": [],
    "reasoning_profile": {
        "global_scores": {},
        "domain_deltas": {},
        "strategy_index": {
            "novel_problem":      ["counterfactual", "hypothesis_competition"],
            "structured_problem": ["decomposition", "causal"],
            "debugging":          ["hypothesis_competition", "discriminating_test"],
        },
        "zpd_ceilings": {},
    },
    "user_preferences": {},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class SelfModel:
    """Manages ``data/self_model.json`` with tamper detection and rollback."""

    def __init__(
        self,
        model_path: str | Path,
        bak_path: str | Path,
        manifest: "StateManifest",
        episodic: "EpisodicMemory",
    ) -> None:
        self._path = Path(model_path)
        self._bak_path = Path(bak_path)
        self._tmp_path = self._path.with_suffix(".json.tmp")
        self._manifest = manifest
        self._episodic = episodic
        self._data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public: lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model, running tamper detection and rollback if needed."""
        stored_hash, stored_bak_hash = self._manifest.read_self_model_hashes()

        # Case A -- first boot
        if stored_hash is None:
            logger.info("SelfModel: first boot -- initialising defaults.")
            self._data = self._deep_copy_defaults()
            self._save_and_update_hashes(audit_kind="self_model_update", reason="first_boot")
            return

        # Main file missing
        if not self._path.exists():
            logger.warning("SelfModel: %s missing -- resetting to defaults.", self._path)
            self._reset_to_defaults(reason="main_file_missing")
            return

        actual_hash = _sha256(self._path)

        if actual_hash == stored_hash:
            # Case B: clean
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                self._ensure_schema()
            except Exception as exc:
                logger.error("SelfModel: failed to parse %s: %s -- resetting.", self._path, exc)
                self._reset_to_defaults(reason="parse_error")
            return

        # Case C: main file hash mismatch
        self._log_violation(
            f"self_model.json hash mismatch (expected {stored_hash[:12]}..., "
            f"got {actual_hash[:12]}...). Checking backup.",
            critical=False,
        )
        # Preserve corrupted copy for forensics
        try:
            shutil.copy2(self._path, self._path.with_name("self_model.corrupted.json"))
        except Exception:
            pass

        if stored_bak_hash is not None and self._bak_path.exists():
            bak_actual_hash = _sha256(self._bak_path)
            if bak_actual_hash == stored_bak_hash:
                # Case C1: backup is clean -- restore
                logger.warning("SelfModel: restoring from backup %s.", self._bak_path)
                try:
                    self._data = json.loads(self._bak_path.read_text(encoding="utf-8"))
                    self._ensure_schema()
                    self._save_and_update_hashes(
                        audit_kind="self_model_update",
                        reason="restored_from_backup",
                    )
                    return
                except Exception as exc:
                    logger.error("SelfModel: backup parse failed: %s -- resetting.", exc)

        # Case C2: backup also corrupted / missing
        self._log_violation(
            "Backup self_model.bak.json is also corrupted or missing. "
            "Resetting to factory defaults.",
            critical=True,
        )
        self._reset_to_defaults(reason="both_files_corrupted")

    # ------------------------------------------------------------------
    # Public: allowed write sources (Mitigation #52)
    # ------------------------------------------------------------------

    def update_zpd_ceilings(self, ceilings: dict[str, int]) -> None:
        """Update ZPD ceilings from a reasoning benchmark run (Phase 4B)."""
        profile = self._data.setdefault("reasoning_profile", {})
        existing = profile.setdefault("zpd_ceilings", {})
        existing.update(ceilings)
        self._save_and_update_hashes(audit_kind="self_model_update", reason="update_zpd_ceilings")

    def update_domain_deltas(self, deltas: dict[str, dict[str, float]]) -> None:
        """Merge Bayesian posterior adjustments from weekly reflection without overwriting global priors."""
        profile = self._data.setdefault("reasoning_profile", {})
        existing = profile.setdefault("domain_deltas", {})
        
        for key, metrics in deltas.items():
            new_ratio = metrics["outcome_ratio"]
            total = metrics["total"]
            
            if key in existing:
                old_ratio = existing[key].get("outcome_ratio", 0.5)
                old_total = existing[key].get("total", 0)
                merged_total = old_total + total
                merged_ratio = ((old_ratio * old_total) + (new_ratio * total)) / merged_total
                existing[key] = {"outcome_ratio": round(merged_ratio, 4), "total": merged_total}
            else:
                existing[key] = {"outcome_ratio": round(new_ratio, 4), "total": total}
                
        self._save_and_update_hashes(audit_kind="self_model_update", reason="update_domain_deltas")

    def increment_boot_count(self) -> None:
        """Called once during agent startup."""
        self._data["boot_count"] = self._data.get("boot_count", 0) + 1
        self._save_and_update_hashes(audit_kind="self_model_update", reason="boot_count")

    def update_competence(self, topic: str, passed: bool, source: str = "benchmark") -> None:
        """Update the empirical competence matrix for *topic*."""
        matrix = self._data.setdefault("empirical_competence_matrix", {})
        entry = matrix.setdefault(topic, {
            "skills_verified": 0,
            "skills_failed": 0,
            "pass_ratio": 0.0,
            "confidence": 0.5,
        })
        if passed:
            entry["skills_verified"] = entry.get("skills_verified", 0) + 1
        else:
            entry["skills_failed"] = entry.get("skills_failed", 0) + 1

        total = entry["skills_verified"] + entry["skills_failed"]
        entry["pass_ratio"] = entry["skills_verified"] / total if total else 0.0
        entry["confidence"] = min(1.0, 0.5 + (total * 0.05))

        self._refresh_strength_gap_lists()
        self._save_and_update_hashes(
            audit_kind="self_model_update",
            reason=f"competence_update:{topic}:passed={passed}:source={source}",
        )

    def record_user_correction(
        self,
        topic: str,
        field: str,
        old_val: Any,
        new_val: Any,
    ) -> None:
        """Record an explicit user correction."""
        matrix = self._data.setdefault("empirical_competence_matrix", {})
        entry = matrix.setdefault(topic, {})
        entry[field] = new_val
        self._save_and_update_hashes(
            audit_kind="self_model_update",
            reason=f"user_correction:{topic}.{field}:{old_val!r}->{new_val!r}",
        )

    # ------------------------------------------------------------------
    # Public: read helpers
    # ------------------------------------------------------------------

    def get_competence(self, topic: str) -> dict | None:
        return self._data.get("empirical_competence_matrix", {}).get(topic)

    def as_summary(self) -> str:
        """Compact competence digest for injection into brain prompts."""
        matrix = self._data.get("empirical_competence_matrix", {})
        if not matrix:
            return "[Self-Model] No empirical competence data yet."
        lines = ["[Self-Model] Empirical competence (topic: pass_ratio, confidence):"]
        for topic, entry in sorted(matrix.items()):
            lines.append(
                f"  {topic}: {entry.get('pass_ratio', 0.0):.0%} "
                f"(n={entry.get('skills_verified', 0) + entry.get('skills_failed', 0)}, "
                f"conf={entry.get('confidence', 0.0):.2f})"
            )
        lines.append(f"  Boot count: {self._data.get('boot_count', 0)}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_and_update_hashes(self, audit_kind: str, reason: str) -> None:
        raw = json.dumps(self._data, indent=2, ensure_ascii=False)

        # 1. Atomic write of main file
        self._tmp_path.write_text(raw, encoding="utf-8")
        os.replace(self._tmp_path, self._path)

        # 2. Backup copy
        shutil.copy2(self._path, self._bak_path)

        # 3. Compute both hashes
        main_hash = _sha256(self._path)
        bak_hash  = _sha256(self._bak_path)

        # 4. Persist hashes to manifest (preserves active_task_hash)
        self._manifest.write_manifest(
            state=None,
            self_model_hash=main_hash,
            self_model_bak_hash=bak_hash,
        )

        # 5. Audit log
        self._audit(audit_kind, reason)

    def _reset_to_defaults(self, reason: str) -> None:
        self._data = self._deep_copy_defaults()
        self._save_and_update_hashes(audit_kind="self_model_update", reason=f"reset:{reason}")

    def _ensure_schema(self) -> None:
        """Backfill keys missing from older model files (forward migration)."""
        for key, val in _DEFAULT_MODEL.items():
            if key not in self._data:
                self._data[key] = val if not isinstance(val, dict) else dict(val)
        profile = self._data.setdefault("reasoning_profile", {})
        for key, val in _DEFAULT_MODEL["reasoning_profile"].items():
            if key not in profile:
                profile[key] = val if not isinstance(val, dict) else dict(val)

    def _refresh_strength_gap_lists(self) -> None:
        matrix = self._data.get("empirical_competence_matrix", {})
        self._data["known_strengths"] = [
            t for t, e in matrix.items() if e.get("pass_ratio", 0.0) >= 0.8
        ]
        self._data["known_knowledge_gaps"] = [
            t for t, e in matrix.items() if e.get("pass_ratio", 0.0) < 0.5
        ]

    def _log_violation(self, message: str, critical: bool) -> None:
        prefix = "CRITICAL" if critical else "WARNING"
        logger.warning("SelfModel [%s]: %s", prefix, message)
        self._audit("security_violation", f"[{prefix}] {message}")

    def _audit(self, kind: str, content: str) -> None:
        try:
            from agent.models import EpisodicLog
            log = EpisodicLog(
                kind=kind,  # type: ignore[arg-type]
                content=f"[SelfModel] {content}",
                outcome="success",
            )
            self._episodic.log_event(log)
        except Exception as exc:
            logger.debug("SelfModel: episodic audit failed (non-fatal): %s", exc)

    @staticmethod
    def _deep_copy_defaults() -> dict[str, Any]:
        import copy
        return copy.deepcopy(_DEFAULT_MODEL)
