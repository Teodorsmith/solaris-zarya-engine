import sys
import subprocess
import logging
from typing import Optional, Any
from agent.sandbox.docker_runner import DockerSandboxRunner
from agent.engine.validator import validate_ast, SecurityError

logger = logging.getLogger(__name__)

class SkillSandbox:
    def __init__(self, docker_runner: Optional[DockerSandboxRunner] = None):
        self.docker_runner = docker_runner or DockerSandboxRunner()

    def run(self, code: str, timeout: float = 5.0, allow_unsafe_host: bool = False) -> dict[str, Any]:
        """Unified entrypoint for sandboxing untrusted code."""
        # Layer 0: Always validate AST first. If this fails, we don't run anything.
        try:
            validate_ast(code)
        except SecurityError as e:
            return {
                "status": "error",
                "error": f"AST Validation blocked execution: {e}",
                "isolated": False
            }

        # Attempt Docker execution
        if self.docker_runner.is_available():
            return self.docker_runner.run(code, timeout=timeout)
            
        # Refuse execution by default if Docker is unavailable
        if not allow_unsafe_host:
            logger.warning("Docker unavailable. Local execution refused by default.")
            return {
                "status": "unavailable",
                "error": "Docker not available; execution refused. Use --unsafe-host to override.",
                "isolated": False
            }
            
        # Fallback to local subprocess (only if explicitly requested)
        logger.warning("Docker sandbox unavailable. Executing locally via OPT-IN allow_unsafe_host flag.")
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=timeout
            )
            return {
                "status": "ok" if result.returncode == 0 else "error",
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
                "returncode": result.returncode,
                "isolated": False
            }
        except subprocess.TimeoutExpired as e:
            return {
                "status": "timeout",
                "error": "Execution exceeded timeout limit",
                "isolated": False,
                "stdout": e.stdout.decode("utf-8", errors="replace") if e.stdout else "",
                "stderr": e.stderr.decode("utf-8", errors="replace") if e.stderr else "",
                "returncode": -1
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "isolated": False
            }
