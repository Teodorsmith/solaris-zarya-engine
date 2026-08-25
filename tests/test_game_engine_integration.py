import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from agent.integrations.blender_mcp import (
    BlenderMCPClient,
    BlenderSecurityError,
)
from agent.integrations.unity_mcp import UnityMCPClient
from agent.engine.vcs_manager import VCSManager


@pytest.fixture
def mock_blender_client():
    return BlenderMCPClient(blender_path=Path("mock_blender.exe"))


@pytest.fixture
def mock_unity_client():
    return UnityMCPClient(
        unity_path=Path("mock_unity.exe"),
        project_path=Path("mock_project"),
    )


# --- Blender AST Sandbox Tests ---
def test_blender_ast_valid(mock_blender_client):
    code = """
import bpy
import bmesh
import mathutils

def create_cube():
    bpy.ops.mesh.primitive_cube_add(
        size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0)
    )
    """
    # Should not raise exception
    mock_blender_client.validate_ast(code)


def test_blender_ast_banned_import(mock_blender_client):
    code = "import os\nos.system('echo hi')"
    with pytest.raises(BlenderSecurityError, match="Banned import: os"):
        mock_blender_client.validate_ast(code)

    code2 = "from subprocess import run"
    with pytest.raises(BlenderSecurityError, match="Banned import: subprocess"):
        mock_blender_client.validate_ast(code2)


def test_blender_ast_banned_primitive(mock_blender_client):
    code = "eval('2 + 2')"
    with pytest.raises(BlenderSecurityError, match="Banned function: eval"):
        mock_blender_client.validate_ast(code)


def test_blender_ast_banned_attribute(mock_blender_client):
    code = "cls = bpy.__class__"
    with pytest.raises(
        BlenderSecurityError, match="Banned attribute: __class__"
    ):
        mock_blender_client.validate_ast(code)


# --- Unity MCP Tests ---
def test_parse_roslyn_errors(mock_unity_client):
    log = """
Assets/Scripts/MyScript.cs(12,34): error CS1061: 'MyScript' does not contain a definition for 'Foo'
Assets/Player.cs(5,1): error CS0246: The type or namespace name 'IUnknown' could not be found
"""
    errors = mock_unity_client.parse_roslyn_errors(log)
    assert len(errors) == 2
    assert errors[0]["file"] == "Assets/Scripts/MyScript.cs"
    assert errors[0]["line"] == "12"
    assert errors[0]["code"] == "CS1061"

    assert errors[1]["file"] == "Assets/Player.cs"
    assert errors[1]["line"] == "5"
    assert errors[1]["code"] == "CS0246"


def test_parse_nunit_xml(mock_unity_client, tmp_path):
    # Create mock XML
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<test-run id="2" testcasecount="2" result="Failed" total="2" passed="1" failed="1" inconclusive="0" skipped="0">
    <test-suite type="Assembly" id="1000" name="Tests.dll">
        <test-case id="1001" name="TestOne" result="Passed" />
        <test-case id="1002" name="TestTwo" result="Failed">
            <failure>
                <message><![CDATA[Expected: true\n  But was:  false]]></message>
                <stack-trace><![CDATA[at TestTwo() in C:/project/Tests.cs:line 42]]></stack-trace>
            </failure>
        </test-case>
    </test-suite>
</test-run>
"""
    xml_path = tmp_path / "results.xml"
    xml_path.write_text(xml_content, encoding="utf-8")

    res = mock_unity_client._parse_nunit_xml(xml_path)

    assert res["passed"] == 1
    assert res["failed"] == 1
    assert len(res["failures"]) == 1
    assert res["failures"][0]["name"] == "TestTwo"
    assert "Expected: true" in res["failures"][0]["message"]


# --- VCS Manager Tests ---
@patch("subprocess.run")
def test_vcs_create_feature_branch(mock_run, tmp_path):
    # Setup mock
    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

    vcs = VCSManager(repo_path=tmp_path)
    branch = vcs.create_feature_branch("Implement jump mechanics")

    assert branch == "ai-feat/implement-jump-mechanics"
    mock_run.assert_called_with(
        ["git", "checkout", "-b", "ai-feat/implement-jump-mechanics"],
        cwd=tmp_path, capture_output=True, text=True
    )


@patch("subprocess.run")
def test_vcs_commit_with_smoke_test_aborts(mock_run, tmp_path, mock_unity_client):
    # Mock git branch --show-current
    mock_run.return_value = Mock(
        returncode=0, stdout="ai-feat/test-branch\n", stderr=""
    )

    vcs = VCSManager(repo_path=tmp_path)

    # Mock unity test failure
    with patch.object(
        mock_unity_client, "run_tests", return_value={"status": "failed"}
    ):
        result = vcs.commit_with_smoke_test(
            "test commit", unity_client=mock_unity_client
        )

        assert result["status"] == "error"
        assert "Commit aborted" in result["error"]

        # Verify git commit was NEVER called
        for call in mock_run.call_args_list:
            args = call[0][0]
            assert "commit" not in args


@patch("subprocess.run")
def test_vcs_commit_success(mock_run, tmp_path, mock_unity_client):
    # Mock subprocess calls
    mock_run.side_effect = [
        Mock(returncode=0, stdout="ai-feat/test-branch\n", stderr=""),
        Mock(returncode=0, stdout="", stderr=""),
        Mock(returncode=0, stdout="[ai-feat/test-branch] test", stderr=""),
    ]

    vcs = VCSManager(repo_path=tmp_path)

    # Mock unity test success
    with patch.object(
        mock_unity_client, "run_tests", return_value={"status": "success"}
    ):
        result = vcs.commit_with_smoke_test(
            "test commit", unity_client=mock_unity_client
        )

        assert result["status"] == "success"
        assert result["commit_output"] == "[ai-feat/test-branch] test"
