import subprocess
import sys
from unittest.mock import patch, Mock

from agent.sandbox.docker_runner import DockerSandboxRunner
from agent.sandbox.runner import SkillSandbox

def test_docker_cli_arguments():
    runner = DockerSandboxRunner()
    runner._is_available = True
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout=b"output", stderr=b"")
        
        runner.run("print('hello')")
        
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        
        assert "docker" in cmd
        assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
        assert "--read-only" in cmd
        assert "--tmpfs" in cmd
        assert any(arg.startswith("--memory=") for arg in cmd)
        assert any(arg.startswith("--cpus=") for arg in cmd)
        assert "--privileged" not in cmd
        assert "-v" not in cmd and "--volume" not in cmd
        
        # Test no env secrets via -e or --env
        assert "-e" not in cmd and "--env" not in cmd

def test_docker_timeout():
    runner = DockerSandboxRunner()
    runner._is_available = True
    
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=5.0, output=b"part", stderr=b"err")
        
        result = runner.run("while True: pass", timeout=5.0)
        
        assert result["status"] == "timeout"
        assert result["isolated"] is True
        assert "Execution exceeded timeout limit" in result["error"]

def test_docker_unavailable_refuses_execution_by_default():
    # When docker is unavailable, it should refuse execution by default
    docker_runner = DockerSandboxRunner()
    docker_runner._is_available = False
    
    sandbox = SkillSandbox(docker_runner)
    
    with patch("subprocess.run") as mock_run:
        result = sandbox.run("print('hello')")
        
        assert result["status"] == "unavailable"
        assert result["isolated"] is False
        assert "execution refused" in result["error"].lower()
        mock_run.assert_not_called()

def test_skill_sandbox_fallback_allowed():
    # When docker is unavailable, it should fall back to local subprocess IF explicitly allowed
    docker_runner = DockerSandboxRunner()
    docker_runner._is_available = False
    
    sandbox = SkillSandbox(docker_runner)
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout=b"fallback", stderr=b"")
        
        result = sandbox.run("print('hello')", allow_unsafe_host=True)
        
        assert result["status"] == "ok"
        assert result["isolated"] is False
        assert mock_run.call_args[0][0][0] == sys.executable

def test_ast_block_before_execution():
    # If AST validation fails, it should NOT run docker or local subprocess
    docker_runner = DockerSandboxRunner()
    docker_runner._is_available = True
    
    sandbox = SkillSandbox(docker_runner)
    
    with patch("subprocess.run") as mock_run:
        # Code that fails AST (import os)
        result = sandbox.run("import os\nos.system('echo hi')")
        
        assert result["status"] == "error"
        assert "AST Validation blocked execution" in result["error"]
        assert result["isolated"] is False
        mock_run.assert_not_called()
