# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Secondary Boot Recovery Signal."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agent.models import TaskState


class StateManifest:
    def __init__(self, manifest_file: str | Path):
        self.manifest_file = Path(manifest_file)
        self.tmp_file = self.manifest_file.with_suffix(".json.tmp")

    def write_manifest(
        self,
        state: TaskState | None,
        self_model_hash: str | None = None,
        self_model_bak_hash: str | None = None,
    ) -> None:
        """Atomically write the manifest.  All three payloads are optional and
        independent: pass only the ones that changed to avoid overwriting fields
        written by another caller in the same boot cycle."""
        existing = self.read_manifest() or {}

        if state is None:
            active_hash = None
        else:
            state_json = state.model_dump_json()
            active_hash = hashlib.sha256(state_json.encode("utf-8")).hexdigest()

        data = {
            "active_task_hash": active_hash,
            # Preserve existing self-model hashes if not being updated
            "self_model_hash": self_model_hash
            if self_model_hash is not None
            else existing.get("self_model_hash"),
            "self_model_bak_hash": self_model_bak_hash
            if self_model_bak_hash is not None
            else existing.get("self_model_bak_hash"),
            "updated_at": self._now(),
        }

        with open(self.tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(self.tmp_file, self.manifest_file)

    def read_manifest(self) -> dict | None:
        if not self.manifest_file.exists():
            return None
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def read_self_model_hashes(self) -> tuple[str | None, str | None]:
        """Return ``(self_model_hash, self_model_bak_hash)`` from the manifest.

        Both values are ``None`` when not yet written (e.g. first boot).
        """
        data = self.read_manifest() or {}
        return data.get("self_model_hash"), data.get("self_model_bak_hash")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
