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

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile

from agent.models import SkillResultSchema

logger = logging.getLogger(__name__)

TIER_1_MODULES = frozenset(
    {
        "json",
        "re",
        "math",
        "typing",
        "dataclasses",
        "datetime",
        "collections",
        "pathlib",
        "shlex",
        "argparse",
        "textwrap",
        "enum",
        "uuid",
        "hashlib",
        "base64",
        "copy",
        "functools",
        "itertools",
        "unittest",
    }
)


class SecurityError(Exception):
    pass


class ASTSecurityScanner(ast.NodeVisitor):
    def __init__(self, skill_name: str | None = None):
        self.errors = []
        self.skill_name = skill_name
        # Built-in functions that cannot be called directly by name
        self.banned_names = {
            "eval",
            "exec",
            "compile",
            "globals",
            "locals",
            "vars",
            "getattr",
            "setattr",
            "delattr",
            "__import__",
            "open",
        }
        # Attribute methods that cannot be called even on objects
        self.banned_func_attrs = {
            "eval",
            "exec",
            "compile",
            "globals",
            "locals",
            "vars",
            "getattr",
            "setattr",
            "delattr",
            "__import__",
        }
        self.banned_attrs = {
            "__subclasses__",
            "__bases__",
            "__mro__",
            "__globals__",
            "__dict__",
            "__class__",
            "__builtins__",
        }

    def _get_root_module(self, module_name: str) -> str:
        return module_name.split(".")[0]

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_mod = self._get_root_module(alias.name)
            if root_mod not in TIER_1_MODULES and root_mod != self.skill_name:
                self.errors.append(f"Importing module '{alias.name}' is forbidden.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module is not None:
            root_mod = self._get_root_module(node.module)
            if root_mod not in TIER_1_MODULES and root_mod != self.skill_name:
                self.errors.append(f"Importing from '{node.module}' is forbidden.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.banned_names:
                self.errors.append(f"Function '{node.func.id}' is forbidden.")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.banned_func_attrs:
                self.errors.append(f"Attribute '{node.func.attr}' is forbidden.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in self.banned_attrs:
            self.errors.append(f"Access to '{node.attr}' is forbidden.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.banned_attrs or node.id in self.banned_names:
            self.errors.append(f"Access to '{node.id}' is forbidden.")
        self.generic_visit(node)


def validate_ast(code: str, skill_name: str | None = None) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityError(f"SyntaxError: {e}")

    scanner = ASTSecurityScanner(skill_name)
    scanner.visit(tree)
    if scanner.errors:
        raise SecurityError("AST Validation Failed:\n" + "\n".join(scanner.errors))


class SkillValidator:
    def __init__(self, timeout_seconds: int = 5):
        self.timeout_seconds = timeout_seconds

    def validate_and_run(
        self, skill_name: str, code: str, test_code: str
    ) -> SkillResultSchema:
        """
        1. Parse and validate AST for both skill and tests.
        2. Write to a temporary directory.
        3. Run unit tests.
        4. (Phase 2 limitation) We don't have real-world input yet, so passing tests is sufficient to "verify" logic.
        Since we need to enforce SkillResultSchema, we generate a small harness to run the entrypoint,
        but in Phase 2 it's just mock verification.
        """
        import shutil

        def get_sandbox_cmd(*args) -> list[str]:
            has_docker = shutil.which("docker") is not None
            if has_docker:
                try:
                    # check if daemon is running
                    proc = subprocess.run(["docker", "info"], capture_output=True, timeout=2)
                    if proc.returncode == 0:
                        return ["docker", "run", "--rm", "--network", "none", "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{tmpdir}:/sandbox", "-w", "/sandbox", "python:3.12-slim", "python"] + list(args)
                except Exception:
                    pass
            return [sys.executable] + list(args)

        validate_ast(code, skill_name=skill_name)
        validate_ast(test_code, skill_name=skill_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_path = os.path.join(tmpdir, f"{skill_name}.py")
            test_path = os.path.join(tmpdir, f"test_{skill_name}.py")

            with open(skill_path, "w", encoding="utf-8") as f:
                f.write(code)

            with open(test_path, "w", encoding="utf-8") as f:
                f.write(f"{code}\n\n{test_code}")

            # 1. Run Unit Tests
            try:
                cmd = get_sandbox_cmd("-m", "unittest", f"test_{skill_name}.py")
                proc = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                if proc.returncode != 0:
                    raise SecurityError(
                        f"Unit tests failed:\n{proc.stderr}\n{proc.stdout}"
                    )
            except subprocess.TimeoutExpired:
                self._kill_tree(proc.pid if "proc" in locals() else None)
                raise SecurityError(
                    f"Test execution timed out after {self.timeout_seconds} seconds."
                )

            # 2. To enforce SkillResultSchema, we would normally run the skill with real input here.
            # For Phase 2, passing the generated tests implies logic is sound, but we still ensure
            # it returns a standard JSON result. We write a minimal harness.
            harness_code = f"""
import json
import traceback
try:
    from {skill_name} import execute
    res = execute()
    print(json.dumps({{"skill_name": "{skill_name}", "status": "ok", "result": res, "errors": []}}))
except Exception as e:
    print(json.dumps({{"skill_name": "{skill_name}", "status": "error", "result": None, "errors": [str(e)]}}))
"""
            harness_path = os.path.join(tmpdir, "run_harness.py")
            with open(harness_path, "w", encoding="utf-8") as f:
                f.write(harness_code)

            try:
                cmd = get_sandbox_cmd("run_harness.py")
                hproc = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )

                # Try to parse the last line as JSON
                lines = [
                    line.strip()
                    for line in hproc.stdout.strip().split("\\n")
                    if line.strip()
                ]
                if not lines:
                    raise SecurityError(
                        f"Harness produced no output. Stderr: {hproc.stderr}"
                    )

                try:
                    result_json = json.loads(lines[-1])
                    return SkillResultSchema(**result_json)
                except json.JSONDecodeError:
                    raise SecurityError(
                        f"Harness did not print valid JSON. Output: {hproc.stdout}"
                    )

            except subprocess.TimeoutExpired:
                self._kill_tree(hproc.pid if "hproc" in locals() else None)
                raise SecurityError(
                    f"Harness execution timed out after {self.timeout_seconds} seconds."
                )

    def _kill_tree(self, pid: int | None):
        if not pid:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True
            )

    def run_saved_skill(self, skill, args_raw: str) -> SkillResultSchema:
        """Run a skill that is already saved in the filesystem (from procedural memory) with arguments."""
        # args_raw should be a JSON string like '{"s1": "foo", "s2": "bar"}'
        if not args_raw.strip():
            args_raw = "{}"

        try:
            # Validate that it's proper JSON before running
            json.loads(args_raw)
        except json.JSONDecodeError as e:
            raise SecurityError(f"Arguments must be valid JSON. Error: {e}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # We copy the skill file to the tmpdir to run it
            skill_path = os.path.join(tmpdir, f"{skill.name}.py")
            with (
                open(skill.file_path, "r", encoding="utf-8") as f_in,
                open(skill_path, "w", encoding="utf-8") as f_out,
            ):
                f_out.write(f_in.read())
                
            import shutil

            def get_sandbox_cmd(*args) -> list[str]:
                has_docker = shutil.which("docker") is not None
                if has_docker:
                    try:
                        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=2)
                        if proc.returncode == 0:
                            return ["docker", "run", "--rm", "--network", "none", "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{tmpdir}:/sandbox", "-w", "/sandbox", "python:3.12-slim", "python"] + list(args)
                    except Exception:
                        pass
                return [sys.executable] + list(args)

            # Now write a harness that calls the skill's execute() method with args_raw
            harness_code = f"""
import json
import traceback
try:
    from {skill.name} import execute
    args = json.loads('''{args_raw}''')
    res = execute(**args)
    print(json.dumps({{"skill_name": "{skill.name}", "status": "ok", "result": res, "errors": []}}))
except Exception as e:
    print(json.dumps({{"skill_name": "{skill.name}", "status": "error", "result": None, "errors": [str(e), traceback.format_exc()]}}))
"""
            harness_path = os.path.join(tmpdir, "run_harness.py")
            with open(harness_path, "w", encoding="utf-8") as f:
                f.write(harness_code)

            try:
                cmd = get_sandbox_cmd("run_harness.py")
                hproc = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )

                lines = [
                    line.strip()
                    for line in hproc.stdout.strip().split("\n")
                    if line.strip()
                ]
                if not lines:
                    raise SecurityError(
                        f"Harness produced no output. Stderr: {hproc.stderr}"
                    )

                try:
                    result_json = json.loads(lines[-1])
                    return SkillResultSchema(**result_json)
                except json.JSONDecodeError:
                    raise SecurityError(
                        f"Harness did not print valid JSON. Output: {hproc.stdout}"
                    )

            except subprocess.TimeoutExpired:
                self._kill_tree(hproc.pid if "hproc" in locals() else None)
                raise SecurityError(
                    f"Harness execution timed out after {self.timeout_seconds} seconds."
                )

    def run_counterfactual_test(self, code: str, edge_case_input: str) -> str:
        """Run a counterfactual variant in a sandbox (gVisor runsc if available)."""
        import shutil

        has_gvisor = shutil.which("runsc") is not None

        if not has_gvisor:
            logger.info(
                "[SANDBOX] gVisor not found on host — executed in standard subprocess sandbox."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "counterfactual.py")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(code)
                f.write("\n\n")
                f.write(f"print({edge_case_input})")

            def get_sandbox_cmd(*args) -> list[str]:
                if has_gvisor:
                    return ["runsc", "exec", sys.executable] + list(args)
                has_docker = shutil.which("docker") is not None
                if has_docker:
                    try:
                        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=2)
                        if proc.returncode == 0:
                            return ["docker", "run", "--rm", "--network", "none", "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{tmpdir}:/sandbox", "-w", "/sandbox", "python:3.12-slim", "python"] + list(args)
                    except Exception:
                        pass
                return [sys.executable] + list(args)
                
            cmd = get_sandbox_cmd("counterfactual.py")

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                if proc.returncode != 0:
                    return f"FAILED: {proc.stderr}"
                return f"SUCCESS: {proc.stdout.strip()}"
            except subprocess.TimeoutExpired:
                self._kill_tree(proc.pid if "proc" in locals() else None)
                return "FAILED: Execution timed out."
