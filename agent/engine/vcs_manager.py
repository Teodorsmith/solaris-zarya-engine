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

import subprocess
import re
from typing import Any
from pathlib import Path

from agent.integrations.unity_mcp import UnityMCPClient


class VCSManagerError(Exception):
    pass


class VCSManager:
    """Manages Git operations with strict regression gates."""

    def __init__(self, repo_path: Path | None = None):
        self.repo_path = repo_path or Path.cwd()

    def create_feature_branch(self, task_name: str) -> str:
        """Create a new isolated feature branch for the agent."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", task_name).strip("-").lower()
        branch_name = f"ai-feat/{slug}"

        try:
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 and "already exists" not in result.stderr:
                raise VCSManagerError(
                    f"Failed to create branch: {result.stderr}"
                )
            elif "already exists" in result.stderr:
                # Just checkout existing
                subprocess.run(
                    ["git", "checkout", branch_name],
                    cwd=self.repo_path,
                    capture_output=True,
                )
            return branch_name
        except FileNotFoundError:
            raise VCSManagerError("Git is not installed or not in PATH.")

    def commit_with_smoke_test(
        self, message: str, unity_client: UnityMCPClient | None = None
    ) -> dict[str, Any]:
        """
        Runs headless Unity smoke tests prior to commit.
        Aborts and reports regression if tests fail.
        """
        # Ensure we are on an ai-feat branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        current_branch = result.stdout.strip()
        if not current_branch.startswith("ai-feat/"):
            return {
                "status": "error",
                "error": (
                    f"Current branch '{current_branch}' is not an "
                    "ai-feat/* branch. Aborting commit."
                )
            }

        # Run Unity Smoke Tests
        if unity_client:
            test_result = unity_client.run_tests(
                test_platform="EditMode", timeout=180.0
            )
            if test_result.get("status") != "success":
                # Abort commit
                return {
                    "status": "error",
                    "error": (
                        "Smoke test failed. Commit aborted to prevent "
                        "regression."
                    ),
                    "details": test_result
                }

        # Tests passed, proceed with commit
        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=self.repo_path,
                check=True,
            )
            commit_result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return {
                "status": "success",
                "message": "Commit successful and smoke-tests passed.",
                "commit_output": commit_result.stdout,
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "error": f"Git commit failed: {e.stderr or e.output}",
            }
