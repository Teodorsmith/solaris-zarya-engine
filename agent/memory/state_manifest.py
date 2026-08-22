# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Secondary Boot Recovery Signal."""
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from agent.models import TaskState

class StateManifest:
    def __init__(self, manifest_file: str | Path):
        self.manifest_file = Path(manifest_file)
        self.tmp_file = self.manifest_file.with_suffix('.json.tmp')
        
    def write_manifest(self, state: TaskState | None) -> None:
        if state is None:
            data = {"active_task_hash": None, "updated_at": self._now()}
        else:
            state_json = state.model_dump_json()
            h = hashlib.sha256(state_json.encode('utf-8')).hexdigest()
            data = {"active_task_hash": h, "updated_at": self._now()}
            
        with open(self.tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(self.tmp_file, self.manifest_file)
        
    def read_manifest(self) -> dict | None:
        if not self.manifest_file.exists():
            return None
        try:
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
