import pytest
from pathlib import Path
from agent.engine.exporter import get_topic_slug, init_markdown_note, append_unit_to_markdown

def test_get_topic_slug():
    assert get_topic_slug("World War 2") == "world_war_2"
    assert get_topic_slug("   React.js!   ") == "react_js"

def test_markdown_export(tmp_path, monkeypatch):
    # Override the hardcoded data/knowledge path to use tmp_path
    def mock_path(*args):
        p = tmp_path / "knowledge"
        p.mkdir(exist_ok=True)
        return p
        
    import agent.engine.exporter
    monkeypatch.setattr(agent.engine.exporter.Path, "__new__", 
                        lambda cls, *args, **kwargs: tmp_path / "knowledge" if args and args[0] == "data/knowledge" else object.__new__(cls))
    
    # Init
    file_path = init_markdown_note("Test Topic", 2, "mock-brain")
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "# Curriculum Research: Test Topic" in content
    assert "mock-brain" in content
    assert "**Planned Units:** 2" in content
    
    # Append Unit
    append_unit_to_markdown(
        topic="Test Topic",
        unit_index=1,
        total_units=2,
        unit_title="Introduction",
        passages=["This is a test passage."],
        facts=[{"statement": "The sky is blue", "confidence": 0.99}],
        sources=["https://example.com"]
    )
    
    content = file_path.read_text(encoding="utf-8")
    assert "## Unit 1/2: Introduction" in content
    assert "### Context & Narrative Summary" in content
    assert "This is a test passage." in content
    assert "### Distilled Atomic Facts" in content
    assert "**[Confidence: 0.99]** The sky is blue" in content
    assert "### Sources Consulted" in content
    assert "<https://example.com>" in content
