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

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error
import json
from pathlib import Path
from typing import Any

from agent.config import get_unity_exe, get_unity_project_path, UNITY_DAEMON_PORT


class UnityMCPClient:
    """Unity MCP bridge for headless testing and daemon communication."""

    def __init__(
        self,
        unity_path: Path | None = None,
        project_path: Path | None = None,
        daemon_port: int | None = None,
    ):
        self.unity_path = unity_path or get_unity_exe()
        if not self.unity_path:
            raise RuntimeError(
                "Unity executable not found. Please set UNITY_EXE in .env "
                "or install Unity Hub."
            )
        
        self.project_path = project_path or get_unity_project_path()
        if not self.project_path:
            raise RuntimeError(
                "Unity project path not found. Please set UNITY_PROJECT_PATH "
                "in .env."
            )
            
        self.daemon_port = daemon_port or UNITY_DAEMON_PORT
        self.verify_project_version()

    def verify_project_version(self) -> None:
        """Parses ProjectSettings/ProjectVersion.txt and compares with Unity executable."""
        pv_txt = self.project_path / "ProjectSettings" / "ProjectVersion.txt"
        if not pv_txt.exists():
            return
            
        content = pv_txt.read_text(encoding="utf-8")
        m = re.search(r"m_EditorVersion:\s*([0-9]+\.[0-9]+)", content)
        if not m:
            return
            
        expected_major_minor = m.group(1)
        exe_name = self.unity_path.name
        
        # We can also attempt to read the actual executable version on Windows
        # For now, we extract version from the path if available, e.g., "2022.3.15f1/Editor/Unity.exe"
        exe_path_str = str(self.unity_path)
        m_exe = re.search(r"([0-9]+\.[0-9]+)", exe_path_str)
        if m_exe:
            actual_major_minor = m_exe.group(1)
            if expected_major_minor != actual_major_minor:
                raise RuntimeError(
                    f"Unity version mismatch! Project expects {expected_major_minor}.* "
                    f"but configured UNITY_EXE is {actual_major_minor}.*. "
                    "Aborting to prevent silent project re-serialization."
                )

    def run_tests(
        self, test_platform: str = "EditMode", timeout: float = 120.0
    ) -> dict[str, Any]:
        """
        Executes Unity headless tests.
        test_platform should be 'EditMode' or 'PlayMode'.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            results_xml = Path(temp_dir) / "results.xml"
            log_txt = Path(temp_dir) / "editor.log"

            cmd = [
                str(self.unity_path),
                "-batchmode",
                "-projectPath", str(self.project_path),
                "-runTests",
                "-testPlatform", test_platform,
                "-testResults", str(results_xml),
                "-logFile", str(log_txt),
                "-quit",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if results_xml.exists():
                    test_results = self._parse_nunit_xml(results_xml)
                else:
                    test_results = {"error": "results.xml was not generated."}

                log_content = ""
                if log_txt.exists():
                    log_content = log_txt.read_text(
                        encoding="utf-8", errors="replace"
                    )

                compiler_errors = self.parse_roslyn_errors(
                    log_content or result.stdout or result.stderr
                )
                
                tests_failed = test_results.get("failed")
                status = "success" if result.returncode == 0 and not tests_failed else "failed"

                return {
                    "status": status,
                    "returncode": result.returncode,
                    "test_results": test_results,
                    "compiler_errors": compiler_errors,
                }

            except subprocess.TimeoutExpired:
                return {
                    "status": "error",
                    "error": f"Unity execution timed out after {timeout}s.",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                }

    def _parse_nunit_xml(self, xml_path: Path) -> dict[str, Any]:
        """Parses Unity NUnit XML test results."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            passed = int(root.attrib.get("passed", 0))
            failed = int(root.attrib.get("failed", 0))
            inconclusive = int(root.attrib.get("inconclusive", 0))
            skipped = int(root.attrib.get("skipped", 0))

            failures = []
            for test_case in root.iter("test-case"):
                if test_case.attrib.get("result") == "Failed":
                    name = test_case.attrib.get("name", "Unknown")
                    failure_node = test_case.find("failure")
                    message = ""
                    stack_trace = ""
                    if failure_node is not None:
                        msg_node = failure_node.find("message")
                        if msg_node is not None and msg_node.text:
                            message = msg_node.text
                        stack_node = failure_node.find("stack-trace")
                        if stack_node is not None and stack_node.text:
                            stack_trace = stack_node.text
                    failures.append({
                        "name": name,
                        "message": message,
                        "stack_trace": stack_trace,
                    })

            return {
                "passed": passed,
                "failed": failed,
                "inconclusive": inconclusive,
                "skipped": skipped,
                "failures": failures,
            }
        except Exception as e:
            return {"error": f"Failed to parse NUnit XML: {e}"}

    def parse_roslyn_errors(self, log_output: str) -> list[dict[str, str]]:
        """
        Regex-extracts error codes (CS0246, CS1061, etc.), file paths,
        line numbers, and messages from Unity compilation logs.
        Matches formats like:
        Assets/Scripts/MyScript.cs(12,34): error CS1061: 'MyScript' does not...
        """
        errors = []
        # Regex to match standard C# compiler error formatting
        pattern = re.compile(
            r"^(?P<file>.+\.cs)\((?P<line>\d+),(?P<col>\d+)\):\s+error\s+"
            r"(?P<code>CS\d+):\s+(?P<message>.*)$",
            re.MULTILINE
        )
        for match in pattern.finditer(log_output):
            errors.append(match.groupdict())
        return errors

    def get_scene_hierarchy(self) -> dict[str, Any]:
        """Queries active HTTP daemon if available."""
        url = f"http://127.0.0.1:{self.daemon_port}/hierarchy"
        req = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status == 200:
                    data = response.read().decode("utf-8")
                    return json.loads(data)
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status}",
                    }
        except urllib.error.URLError:
            # Fallback if daemon is not running
            return {
                "status": "fallback",
                "message": (
                    "Unity HTTP daemon not reachable on port "
                    f"{self.daemon_port}. Returning gracefully."
                )
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
