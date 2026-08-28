# Solaris Zarya Engine
# Copyright (C) 2026 Teodor Smith <teosmith.studios@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For commercial licensing options without AGPLv3 network-copyleft obligations,
# contact: teosmith.studios@gmail.com

"""Unity CLI Client — interfaces with the official ``unity`` CLI binary
and ``com.unity.pipeline`` package.

The official Unity CLI turns a running Unity Editor into a local API server.
This client calls into that server via the ``unity`` subprocess with structured
JSON output, replacing the legacy ``UnityMCPClient`` batchmode approach.

Requires:
    - ``unity`` CLI binary installed and on PATH (see Unity CLI docs).
    - ``com.unity.pipeline`` package added to the target Unity project.

Falls back gracefully when the CLI is unavailable, allowing the rest of the
agent to function without Unity.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class UnityCLINotFoundError(RuntimeError):
    """Raised when the ``unity`` CLI binary is not on PATH."""


class UnityCLIClient:
    """Interfaces with Unity Editor instances via the official Unity CLI.

    The official CLI provides deterministic, structured commands that work
    even during domain reloads and compiles.  Each call returns JSON when
    ``--json`` is passed, giving the agent clean, parseable results.

    Key capabilities used:
        * ``unity command run-tests``  — run NUnit tests with JSON results
        * ``unity command eval``       — execute C# inside the live editor
        * ``unity status``             — check connected editor state
        * ``unity command list``       — list available project commands
    """

    def __init__(self, project_path: str | Path | None = None) -> None:
        self._cli_path = shutil.which("unity")
        if not self._cli_path:
            raise UnityCLINotFoundError(
                "Unity CLI is not installed or not on PATH. "
                "Install it from the Unity CLI docs page, then ensure "
                "'unity --version' works in your terminal."
            )
        self.project_path = str(Path(project_path).resolve()) if project_path else None

    # ------------------------------------------------------------------
    # Core subprocess wrapper
    # ------------------------------------------------------------------

    def _run_cli(
        self,
        args: list[str],
        *,
        timeout: float = 120.0,
        cwd: str | None = None,
    ) -> tuple[int, dict[str, Any] | str]:
        """Run a ``unity`` CLI command and return ``(exit_code, parsed_output)``.

        Appends ``--json`` and ``--non-interactive`` automatically so every
        call produces structured output with no TTY prompts.
        """
        cmd = [self._cli_path, *args, "--json", "--non-interactive"]
        work_dir = cwd or self.project_path

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                cwd=work_dir,
            )

            # Try to parse structured JSON from stdout
            stdout = result.stdout.strip()
            try:
                data: dict[str, Any] | str = json.loads(stdout) if stdout else {}
            except json.JSONDecodeError:
                # CLI printed non-JSON (e.g. plain text status)
                data = stdout

            if result.returncode != 0 and isinstance(data, dict):
                # Enrich with stderr if present
                if result.stderr.strip():
                    data.setdefault("stderr", result.stderr.strip())

            return result.returncode, data

        except subprocess.TimeoutExpired:
            logger.error("Unity CLI command timed out after %ss: %s", timeout, cmd)
            return -1, {"error": f"Unity CLI timed out after {timeout}s"}
        except FileNotFoundError:
            logger.error("Unity CLI binary not found at %s", self._cli_path)
            return -1, {"error": "unity_cli_not_found"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return the live state of connected Unity Editor instances."""
        code, data = self._run_cli(["status"])
        if isinstance(data, dict):
            data["exit_code"] = code
            return data
        return {"raw": data, "exit_code": code}

    def list_commands(self) -> dict[str, Any]:
        """List all available project commands exposed by ``com.unity.pipeline``."""
        code, data = self._run_cli(["command", "list"])
        if isinstance(data, dict):
            data["exit_code"] = code
            return data
        return {"raw": data, "exit_code": code}

    def eval_csharp(self, code_snippet: str) -> dict[str, Any]:
        """Execute arbitrary C# inside the running Unity Editor.

        .. warning::
            This runs code with full editor privileges.  The agent's
            Governor / HITL gate **must** approve the code before calling
            this method.
        """
        code, data = self._run_cli(["command", "eval", code_snippet])
        if code != 0:
            logger.error("Unity C# eval failed (exit %d): %s", code, data)
        if isinstance(data, dict):
            data["exit_code"] = code
            data["status"] = "success" if code == 0 else "error"
            return data
        return {"raw": data, "exit_code": code, "status": "success" if code == 0 else "error"}

    def run_tests(
        self,
        test_platform: str = "EditMode",
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Run NUnit tests via the official CLI and return structured results.

        Returns a dict with at least:
            ``status``  — ``"success"`` or ``"failed"`` or ``"error"``
            ``test_results`` — parsed JSON from the CLI
            ``compiler_errors`` — list (may be empty)
        """
        code, data = self._run_cli(
            ["test", "--test-platform", test_platform],
            timeout=timeout,
        )

        # Exit code 0 = all tests pass
        # Exit code 2 = tests ran but some failed
        # Exit code 6 = tests ran with failures (Unity convention)
        # Other non-zero = infrastructure error

        if isinstance(data, dict):
            # Structured JSON from CLI
            passed = data.get("passed", 0)
            failed = data.get("failed", 0)
            failures = data.get("failures", [])
            compiler_errors = data.get("compiler_errors", [])

            status = "success" if code == 0 and failed == 0 else "failed"
            if code not in (0, 2, 6):
                status = "error"

            result = {
                "status": status,
                "returncode": code,
                "test_results": {
                    "passed": passed,
                    "failed": failed,
                    "failures": failures,
                },
                "compiler_errors": compiler_errors,
            }
            # Propagate error message from _run_cli (e.g. timeout)
            if "error" in data:
                result["error"] = data["error"]
            return result
        else:
            # Plain text fallback — likely an error message
            return {
                "status": "error",
                "returncode": code,
                "error": data if data else "No output from Unity CLI",
                "test_results": {},
                "compiler_errors": [],
            }

    def recompile(self, timeout: float = 60.0) -> dict[str, Any]:
        """Trigger a script recompilation in the connected editor."""
        code, data = self._run_cli(["command", "recompile"], timeout=timeout)
        if isinstance(data, dict):
            data["exit_code"] = code
            return data
        return {"raw": data, "exit_code": code}

    def enter_play_mode(self) -> dict[str, Any]:
        """Enter Play Mode in the connected editor."""
        code, data = self._run_cli(["command", "play"])
        if isinstance(data, dict):
            data["exit_code"] = code
            return data
        return {"raw": data, "exit_code": code}

    def install_pipeline(self, project_path: str | Path | None = None) -> dict[str, Any]:
        """Install the ``com.unity.pipeline`` package into a Unity project.

        This is a convenience method for first-time setup.
        """
        target = str(Path(project_path).resolve()) if project_path else self.project_path
        if not target:
            return {"error": "No project path specified"}
        code, data = self._run_cli(["pipeline", "install", target])
        if isinstance(data, dict):
            data["exit_code"] = code
            return data
        return {"raw": data, "exit_code": code}
