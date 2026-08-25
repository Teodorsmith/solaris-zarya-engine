import subprocess
import logging
from typing import Optional, Any
from agent.config import DOCKER_SANDBOX_IMAGE, DOCKER_SANDBOX_MEMORY, DOCKER_SANDBOX_CPUS

logger = logging.getLogger(__name__)

class DockerSandboxRunner:
    def __init__(self):
        self._is_available = None

    def is_available(self) -> bool:
        """Verifies Docker daemon connectivity."""
        if self._is_available is not None:
            return self._is_available
            
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=2.0
            )
            self._is_available = (result.returncode == 0)
        except (subprocess.SubprocessError, FileNotFoundError):
            self._is_available = False
            
        if not self._is_available:
            logger.warning("Docker daemon is offline or missing. Sandbox runner unavailable.")
        return self._is_available

    def run(self, code: str, timeout: Optional[float] = None) -> dict[str, Any]:
        """Executes untrusted code within an isolated Docker container."""
        if not self.is_available():
            return {"status": "error", "error": "Docker not available"}

        cmd = [
            "docker", "run", "--rm", "-i",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            f"--memory={DOCKER_SANDBOX_MEMORY}",
            f"--cpus={DOCKER_SANDBOX_CPUS}",
            "--user", "1000:1000",
            # Additional safety from review
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "64",
            DOCKER_SANDBOX_IMAGE,
            "python", "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                input=code.encode("utf-8"),
                capture_output=True,
                timeout=timeout
            )
            return {
                "status": "ok" if result.returncode == 0 else "error",
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
                "returncode": result.returncode,
                "isolated": True
            }
        except subprocess.TimeoutExpired as e:
            return {
                "status": "timeout",
                "error": "Execution exceeded timeout limit",
                "isolated": True,
                "stdout": e.stdout.decode("utf-8", errors="replace") if e.stdout else "",
                "stderr": e.stderr.decode("utf-8", errors="replace") if e.stderr else "",
                "returncode": -1
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "isolated": True
            }
