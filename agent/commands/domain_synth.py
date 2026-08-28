# Solaris Zarya Engine
# Copyright (C) 2026 Teodor Smith <teosmith.studios@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import logging
import uuid
import time
from pathlib import Path
from rich.console import Console

from agent.engine.synthesizer import SkillSynthesizer, SynthesizerError
from agent.engine.vcs_manager import VCSManager
from agent.engine.state_machine import TaskFSM
from agent.engine.governor import PermissionGovernor
from agent.models import EpisodicLog

console = Console()
logger = logging.getLogger(__name__)


def _create_unity_client():
    """Prefer the official Unity CLI; fall back to legacy batchmode client."""
    try:
        from agent.integrations.unity_cli import UnityCLIClient
        client = UnityCLIClient()
        logger.info("Using official Unity CLI client.")
        return client
    except Exception:
        pass

    try:
        from agent.integrations.unity_mcp import UnityMCPClient
        client = UnityMCPClient()
        logger.info("Unity CLI not available — falling back to legacy UnityMCPClient (batchmode).")
        return client
    except Exception as e:
        raise RuntimeError(
            f"Neither Unity CLI nor legacy UnityMCPClient could be initialised: {e}"
        ) from e


def handle_unity_synth(
    topic: str,
    synthesizer: SkillSynthesizer,
    vcs_manager: VCSManager,
    fsm: TaskFSM,
    governor: PermissionGovernor,
    episodic_memory
) -> None:
    if not topic:
        console.print("[yellow]usage: unity-synth <topic>[/yellow]")
        return
        
    try:
        unity_client = _create_unity_client()
    except Exception as e:
        console.print(f"[red]Failed to initialize Unity client: {e}[/red]")
        return
        
    task_id = f"unity_synth_{uuid.uuid4().hex[:8]}"
    console.print(f"Starting Unity C# Synthesis Task: {task_id} - '{topic}'")
    
    # 1. Initialize FSM
    fsm.start_task(goal_id=task_id)
    if episodic_memory:
        episodic_memory.log_event(EpisodicLog(
            trace_id=task_id,
            kind="system",
            content=f"Started Unity synthesis for topic: {topic}",
            outcome="neutral"
        ))
        
    # 2. Worktree Isolation
    try:
        worktree_path = vcs_manager.create_staging_worktree(topic)
        branch_name = f"ai-feat/{synthesizer._slugify_topic(topic)}"
        console.print(f"Created isolated staging worktree: {worktree_path}")
    except Exception as e:
        console.print(f"[red]Failed to create VCS worktree: {e}[/red]")
        fsm.advance("FAILED")
        return

    # 3. Synthesis & Headless Run
    fsm.advance("RUNNING")
    try:
        console.print("Synthesizing and testing C# script...")
        skill = synthesizer.synthesize_csharp_script(
            topic=topic,
            context="Autonomous Unity C# generation",
            unity_client=unity_client,
            staging_path=worktree_path
        )
    except SynthesizerError as e:
        console.print(f"[red]Synthesis failed: {e}[/red]")
        _rollback(fsm, vcs_manager, worktree_path, branch_name, episodic_memory, task_id)
        return
        
    # 4. Governor Tier-2 Gate
    fsm.advance("VERIFYING")
    approved = governor.request_skill_write_permission(
        skill_name=skill.name,
        file_path="Assets/Scripts/" + skill.name + ".cs",
        code="<C# script verified in staging worktree>"
    )
    
    if not approved:
        console.print("[yellow]Merge rejected by Governor. Rolling back.[/yellow]")
        _rollback(fsm, vcs_manager, worktree_path, branch_name, episodic_memory, task_id)
        return
        
    # 5. Commit & Cleanup
    try:
        console.print(f"[green]Merging {branch_name} into main workspace...[/green]")
        vcs_manager.cleanup_worktree(worktree_path, branch_name, delete_branch=False)
        # Merge branch (simulated standard git flow)
        import subprocess
        subprocess.run(["git", "merge", "--no-ff", branch_name, "-m", f"Merge synthesized C# skill: {topic}"], cwd=vcs_manager.repo_path)
        subprocess.run(["git", "branch", "-d", branch_name], cwd=vcs_manager.repo_path)
        fsm.advance("COMPLETED")
        if episodic_memory:
            episodic_memory.log_event(EpisodicLog(
                trace_id=task_id, kind="system", content=f"Successfully synthesized and merged Unity skill: {skill.name}", outcome="success"
            ))
        console.print(f"[bold green]Successfully delivered Unity C# Skill: {skill.name}[/bold green]")
    except Exception as e:
        console.print(f"[red]Failed during merge/cleanup: {e}[/red]")
        fsm.advance("FAILED")


def handle_blender_synth(
    topic: str,
    synthesizer: SkillSynthesizer,
    vcs_manager: VCSManager,
    fsm: TaskFSM,
    governor: PermissionGovernor,
    episodic_memory
) -> None:
    if not topic:
        console.print("[yellow]usage: blender-synth <topic>[/yellow]")
        return
        
    try:
        from agent.integrations.blender_mcp import BlenderMCPClient
        blender_client = BlenderMCPClient()
    except Exception as e:
        console.print(f"[red]Failed to initialize BlenderMCPClient: {e}[/red]")
        return
        
    task_id = f"blender_synth_{uuid.uuid4().hex[:8]}"
    console.print(f"Starting Blender Python Synthesis Task: {task_id} - '{topic}'")
    
    # 1. Initialize FSM
    fsm.start_task(goal_id=task_id)
    if episodic_memory:
        episodic_memory.log_event(EpisodicLog(
            trace_id=task_id, kind="system", content=f"Started Blender synthesis for topic: {topic}", outcome="neutral"
        ))
        
    # 2. Staging Export Dir
    export_dir = Path("data/exports") / f"blender_{uuid.uuid4().hex[:6]}"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Synthesis & Headless Run
    fsm.advance("RUNNING")
    try:
        console.print("Synthesizing and testing Blender script...")
        skill = synthesizer.synthesize_blender_script(
            topic=topic,
            context="Autonomous Blender generation",
            blender_client=blender_client,
            export_dir=export_dir
        )
    except SynthesizerError as e:
        console.print(f"[red]Synthesis failed: {e}[/red]")
        fsm.advance("FAILED")
        if episodic_memory:
            episodic_memory.log_event(EpisodicLog(trace_id=task_id, kind="system", content=f"Blender Synthesis Failed: {e}", outcome="failure"))
        return
        
    # 4. Governor Tier-2 Gate for Exporting Assets
    fsm.advance("VERIFYING")
    approved = governor.request_skill_write_permission(
        skill_name=skill.name,
        file_path="Assets/Models/...",
        code=f"<Blender script validated. Assets staged at {export_dir}>"
    )
    
    if not approved:
        console.print("[yellow]Asset export rejected by Governor.[/yellow]")
        fsm.advance("CANCELLED")
        return
        
    console.print(f"[green]Blender synthesis complete. Staged assets at {export_dir}[/green]")
    fsm.advance("COMPLETED")
    if episodic_memory:
        episodic_memory.log_event(EpisodicLog(
            trace_id=task_id, kind="system", content=f"Successfully synthesized Blender skill: {skill.name}", outcome="success"
        ))

def _rollback(fsm, vcs_manager, worktree_path, branch_name, episodic_memory, task_id):
    try:
        vcs_manager.cleanup_worktree(worktree_path, branch_name, delete_branch=True)
        console.print(f"[green]Atomic rollback complete: removed worktree {worktree_path} and branch {branch_name}[/green]")
    except Exception as cleanup_err:
        console.print(f"[red]Rollback partial failure: {cleanup_err}[/red]")
    fsm.advance("FAILED")
    if episodic_memory:
        episodic_memory.log_event(EpisodicLog(trace_id=task_id, kind="system", content=f"Task Failed and Rolled Back", outcome="failure"))
