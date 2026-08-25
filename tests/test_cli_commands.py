import pytest
from unittest.mock import patch

@pytest.fixture
def capture_console(monkeypatch):
    from rich.console import Console

    console = Console(record=True)
    import agent.cli

    monkeypatch.setattr(agent.cli, "console", console)
    return console


def test_skills_command_fuzzy_search(capture_console, tmp_path):
    # Setup procedural memory with two skills
    from agent.memory.procedural import ProceduralMemory
    from agent.models import Skill

    db_path = tmp_path / "skills.db"
    procedural = ProceduralMemory(db_path)

    # Register skills
    skill1 = Skill(
        name="levenshtein_distance",
        description="Calculates edit distance between two strings",
        file_path="skills/levenshtein.py",
        verification_tier="real_local",
    )
    skill2 = Skill(
        name="json_formatter",
        description="Formats JSON text",
        file_path="skills/json_format.py",
        verification_tier="mock",
    )
    procedural.register(skill1)
    procedural.register(skill2)

    from agent.commands.skills import handle_skills

    # 1. Empty query should list all
    with patch("agent.commands.skills.console.print") as mock_print:
        handle_skills("", procedural)
        mock_print.assert_called()

    # 2. Test valid fuzzy search matches name
    with patch("agent.commands.skills.console.print") as mock_print:
        handle_skills("levenshtein", procedural)
        args, _ = mock_print.call_args
        table_str = str(args[0].__dict__) if args else ""
        assert "levenshtein_distance" in table_str
        assert "json_formatter" not in table_str

    # 3. Test valid fuzzy search matches description
    with patch("agent.commands.skills.console.print") as mock_print:
        handle_skills("edit distance", procedural)
        args, _ = mock_print.call_args
        table_str = str(args[0].__dict__) if args else ""
        assert "levenshtein_distance" in table_str
        assert "json_formatter" not in table_str

    # 4. Test nonexistent
    with patch("agent.commands.skills.console.print") as mock_print:
        handle_skills("nonexistent", procedural)
        args, _ = mock_print.call_args
        assert "No skills found matching 'nonexistent'" in args[0]


def test_read_command(capture_console, tmp_path):
    # Create test markdown file
    md_file = tmp_path / "test_topic.md"
    md_file.write_text("# Hello World\nThis is a test.", encoding="utf-8")

    from agent.commands.system import handle_read_file

    with patch("agent.commands.system.console.print") as mock_print:
        # Valid file
        handle_read_file(str(md_file))
        mock_print.assert_called()

        # Missing file
        handle_read_file("nonexistent.md")
        mock_print.assert_called_with("[red]File not found: nonexistent.md[/]")


def test_brain_command_list_and_switch(capture_console):
    from agent.brains.factory import BrainManager
    from agent.brains.mock_brain import MockBrain
    from agent.commands.system import handle_brain_cmd

    brain_manager = BrainManager()
    
    with patch("agent.commands.system.console.print") as mock_print:
        # Test brain list
        handle_brain_cmd("list", brain_manager)
        
        table_output = None
        for call_args, _ in mock_print.call_args_list:
            if hasattr(call_args[0], "columns"):
                table_output = call_args[0]
                break
                
        assert table_output is not None, "Expected rich Table to be printed"
        # Verify the active brain is printed correctly without the curly brace literal
        active_found = False
        for call_args, _ in mock_print.call_args_list:
            if "Active brain: [bold cyan]MockBrain[/bold cyan]" in str(call_args[0]):
                active_found = True
        assert active_found, "Did not find properly evaluated current active brain"

        # Test brain switch
        mock_print.reset_mock()
        handle_brain_cmd("switch mock", brain_manager)
        
        switch_found = False
        for call_args, _ in mock_print.call_args_list:
            if "Switched brain: MockBrain -> MockBrain" in str(call_args[0]):
                switch_found = True
        assert switch_found, "Did not find expected switch message"
