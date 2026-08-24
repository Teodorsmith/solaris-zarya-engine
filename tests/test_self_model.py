# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for agent/memory/self_model.py (Phase 4A)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_env(tmp: Path):
    """Build a SelfModel with a real episodic DB and manifest in *tmp*."""
    import sqlite3
    from agent.memory.state_manifest import StateManifest
    from agent.memory.episodic import EpisodicMemory
    from agent.memory.self_model import SelfModel

    model_path = tmp / "self_model.json"
    bak_path   = tmp / "self_model.bak.json"
    manifest_path = tmp / "state_manifest.json"
    episodic_path = tmp / "episodic.db"

    episodic = EpisodicMemory(episodic_path)
    manifest = StateManifest(manifest_path)
    sm = SelfModel(model_path, bak_path, manifest, episodic)
    return sm, manifest, episodic, model_path, bak_path, manifest_path


class TestSelfModelDefaults(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_load_defaults(self):
        sm, manifest, episodic, model_path, bak_path, _ = _make_env(self._tmp)
        sm.load()
        data = sm.as_dict()
        self.assertEqual(data["boot_count"], 0)
        self.assertIn("empirical_competence_matrix", data)
        self.assertIn("reasoning_profile", data)
        self.assertTrue(model_path.exists())
        self.assertTrue(bak_path.exists())

    def test_atomic_save_roundtrip(self):
        sm, _, _, model_path, _, _ = _make_env(self._tmp)
        sm.load()
        sm.increment_boot_count()
        # Reload fresh instance from same files
        from agent.memory.state_manifest import StateManifest
        from agent.memory.episodic import EpisodicMemory
        from agent.memory.self_model import SelfModel
        manifest2 = StateManifest(self._tmp / "state_manifest.json")
        sm2 = SelfModel(model_path, self._tmp / "self_model.bak.json", manifest2,
                        EpisodicMemory(self._tmp / "episodic.db"))
        sm2.load()
        self.assertEqual(sm2.as_dict()["boot_count"], 1)

    def test_backup_hash_stored(self):
        sm, manifest, _, model_path, bak_path, _ = _make_env(self._tmp)
        sm.load()
        _, stored_bak_hash = manifest.read_self_model_hashes()
        actual_bak_hash = _sha256(bak_path)
        self.assertEqual(stored_bak_hash, actual_bak_hash)

    def test_update_competence_pass(self):
        sm, _, _, _, _, _ = _make_env(self._tmp)
        sm.load()
        sm.update_competence("git", passed=True)
        entry = sm.get_competence("git")
        self.assertEqual(entry["skills_verified"], 1)
        self.assertEqual(entry["skills_failed"], 0)
        self.assertAlmostEqual(entry["pass_ratio"], 1.0)

    def test_update_competence_fail(self):
        sm, _, _, _, _, _ = _make_env(self._tmp)
        sm.load()
        sm.update_competence("docker", passed=False)
        entry = sm.get_competence("docker")
        self.assertEqual(entry["skills_failed"], 1)
        self.assertAlmostEqual(entry["pass_ratio"], 0.0)

    def test_write_protection(self):
        """Direct dict mutation does not persist."""
        sm, _, _, model_path, _, _ = _make_env(self._tmp)
        sm.load()
        # Mutate internal data directly (simulating a bug, not an allowed path)
        sm._data["boot_count"] = 999
        # Do NOT call any save method — reload and check
        from agent.memory.state_manifest import StateManifest
        from agent.memory.episodic import EpisodicMemory
        from agent.memory.self_model import SelfModel
        manifest2 = StateManifest(self._tmp / "state_manifest.json")
        sm2 = SelfModel(model_path, self._tmp / "self_model.bak.json", manifest2,
                        EpisodicMemory(self._tmp / "episodic.db"))
        sm2.load()
        self.assertNotEqual(sm2.as_dict()["boot_count"], 999)

    def test_audit_log(self):
        sm, _, episodic, _, _, _ = _make_env(self._tmp)
        sm.load()  # first_boot -> audit
        sm.increment_boot_count()  # boot_count -> audit
        logs = episodic.recent(10)
        kinds = [l.kind for l in logs]
        self.assertIn("self_model_update", kinds)


class TestSelfModelTamper(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _boot_fresh(self):
        sm, manifest, episodic, model_path, bak_path, manifest_path = _make_env(self._tmp)
        sm.load()
        sm.increment_boot_count()
        return model_path, bak_path, manifest_path

    def test_tamper_main_restores_from_backup(self):
        """Corrupt main -> backup clean -> restored."""
        model_path, bak_path, manifest_path = self._boot_fresh()
        # Corrupt only the main file
        model_path.write_text('{"boot_count": 9999}', encoding="utf-8")

        from agent.memory.state_manifest import StateManifest
        from agent.memory.episodic import EpisodicMemory
        from agent.memory.self_model import SelfModel
        manifest2 = StateManifest(manifest_path)
        episodic2 = EpisodicMemory(self._tmp / "episodic.db")
        sm2 = SelfModel(model_path, bak_path, manifest2, episodic2)
        sm2.load()

        # Boot count should be 1 (from backup), not 9999
        self.assertEqual(sm2.as_dict()["boot_count"], 1)

        # security_violation should be in episodic
        kinds = [l.kind for l in episodic2.recent(20)]
        self.assertIn("security_violation", kinds)

        # Both hashes should be consistent after restore
        stored_main, stored_bak = manifest2.read_self_model_hashes()
        self.assertEqual(stored_main, _sha256(model_path))
        self.assertEqual(stored_bak, _sha256(bak_path))

    def test_tamper_both_files_resets_to_defaults(self):
        """Both files corrupt -> factory reset."""
        model_path, bak_path, manifest_path = self._boot_fresh()
        model_path.write_text('{"boot_count": 9999}', encoding="utf-8")
        bak_path.write_text('{"boot_count": 8888}', encoding="utf-8")

        from agent.memory.state_manifest import StateManifest
        from agent.memory.episodic import EpisodicMemory
        from agent.memory.self_model import SelfModel
        manifest2 = StateManifest(manifest_path)
        episodic2 = EpisodicMemory(self._tmp / "episodic.db")
        sm2 = SelfModel(model_path, bak_path, manifest2, episodic2)
        sm2.load()

        # Reset to defaults -> boot_count = 0
        self.assertEqual(sm2.as_dict()["boot_count"], 0)

        # CRITICAL violation logged
        logs = episodic2.recent(30)
        violations = [l for l in logs if l.kind == "security_violation" and "CRITICAL" in l.content]
        self.assertTrue(len(violations) >= 1)

    def test_tamper_no_backup_resets_to_defaults(self):
        """Main corrupt + no backup -> factory reset."""
        model_path, bak_path, manifest_path = self._boot_fresh()
        model_path.write_text('{"boot_count": 9999}', encoding="utf-8")
        bak_path.unlink()

        from agent.memory.state_manifest import StateManifest
        from agent.memory.episodic import EpisodicMemory
        from agent.memory.self_model import SelfModel
        manifest2 = StateManifest(manifest_path)
        episodic2 = EpisodicMemory(self._tmp / "episodic.db")
        sm2 = SelfModel(model_path, bak_path, manifest2, episodic2)
        sm2.load()

        self.assertEqual(sm2.as_dict()["boot_count"], 0)


if __name__ == "__main__":
    unittest.main()
