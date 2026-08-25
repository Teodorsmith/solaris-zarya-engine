import ast
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent.config import get_blender_exe, get_unity_project_path


class BlenderSecurityError(Exception):
    pass


class BlenderMCPClient:
    """Headless Blender execution bridge with AST security constraints."""

    ALLOWED_IMPORTS = {"bpy", "bmesh", "mathutils", "math", "pathlib"}
    BANNED_IMPORTS = {
        "os", "sys", "subprocess", "socket", "ctypes", "importlib",
    }
    BANNED_PRIMITIVES = {"eval", "exec", "__import__"}
    BANNED_ATTRIBUTES = {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__globals__",
        "__dict__",
        "__builtins__",
    }

    def __init__(self, blender_path: Path | None = None):
        self.blender_path = blender_path or get_blender_exe()
        if not self.blender_path:
            raise RuntimeError(
                "Blender executable not found. Please set BLENDER_EXE in .env "
                "or install Blender and add it to PATH."
            )

    def validate_ast(self, code: str) -> None:
        """Validate Blender Python script against strict security rules."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise BlenderSecurityError(f"Syntax error: {e}")

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in self.BANNED_IMPORTS:
                        raise BlenderSecurityError(f"Banned import: {name}")
                    if name not in self.ALLOWED_IMPORTS:
                        raise BlenderSecurityError(
                            f"Disallowed import: {name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    name = node.module.split(".")[0]
                    if name in self.BANNED_IMPORTS:
                        raise BlenderSecurityError(f"Banned import: {name}")
                    if name not in self.ALLOWED_IMPORTS:
                        raise BlenderSecurityError(
                            f"Disallowed import: {name}"
                        )

            # Check for banned primitives (eval, exec, __import__)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BANNED_PRIMITIVES:
                        raise BlenderSecurityError(
                            f"Banned function: {node.func.id}"
                        )

            # Check for banned attributes
            elif isinstance(node, ast.Attribute):
                if node.attr in self.BANNED_ATTRIBUTES:
                    raise BlenderSecurityError(
                        f"Banned attribute: {node.attr}"
                    )
                if (
                    node.attr == "modules"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "sys"
                ):
                    raise BlenderSecurityError("Banned attribute: sys.modules")

    def run_script(
        self, script_code: str, timeout: float = 60.0
    ) -> dict[str, Any]:
        """Execute a Blender python script headlessly."""
        self.validate_ast(script_code)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "script.py"
            temp_file.write_text(script_code, encoding="utf-8")

            cmd = [
                str(self.blender_path),
                "--background",
                "--factory-startup",
                "--python", str(temp_file),
                "--python-exit-code", "1",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=temp_dir,
                )

                if result.returncode == 0:
                    return {
                        "status": "success",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                else:
                    error_msg = self._extract_blender_error(
                        result.stderr or result.stdout
                    )
                    return {
                        "status": "error",
                        "error": error_msg,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
            except subprocess.TimeoutExpired:
                return {
                    "status": "error",
                    "error": f"Blender script timed out after {timeout}s.",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                }

    def _extract_blender_error(self, output: str) -> str:
        """Extract the python traceback from Blender output."""
        match = re.search(
            r"(Traceback \(most recent call last\):.*)",
            output,
            flags=re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return "Unknown error. Check stdout/stderr."

    def generate_mesh(
        self,
        procedural_code: str,
        output_path: Path,
        export_format: str = "gltf",
    ) -> dict[str, Any]:
        """
        Clears scene, runs mesh code, applies auto-UV unwrapping,
        and exports .gltf or .fbx.
        Output path is enforced to be within the active project root.
        """
        project_root = get_unity_project_path()
        if project_root is None:
            # For testing without unity context
            pass
        else:
            try:
                output_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                return {
                    "status": "error",
                    "error": "Output path must be inside Unity project root.",
                }

        # Build the wrapper script
        wrapper = f"""
import bpy
import bmesh
import math
import mathutils
from pathlib import Path

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Run user code
{procedural_code}

# Auto-UV unwrap all meshes
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)

# Export
output_file = r"{str(output_path)}"
export_format = "{export_format.lower()}"

if export_format == "fbx":
    bpy.ops.export_scene.fbx(filepath=output_file, use_selection=False)
elif export_format in ("gltf", "glb"):
    bpy.ops.export_scene.gltf(filepath=output_file, use_selection=False)
else:
    raise ValueError(f"Unsupported export format: {{export_format}}")
"""
        return self.run_script(wrapper, timeout=120.0)
