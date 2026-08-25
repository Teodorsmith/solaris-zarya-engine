import os
from pathlib import Path
from agent.engine.exporter import (
    append_unit_to_markdown,
    get_topic_slug,
    init_markdown_note,
)


def test_get_topic_slug():
    assert get_topic_slug("World War 2") == "world_war_2"
    assert get_topic_slug("   React.js!   ") == "react_js"


def test_markdown_export(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(exist_ok=True)

    import agent.engine.exporter

    def mock_path(path_str):
        if path_str == "data/knowledge":
            return knowledge_dir
        return Path(path_str)

    monkeypatch.setattr(agent.engine.exporter, "Path", mock_path)

    # Init
    file_path = init_markdown_note("Test Topic", 2, "mock-brain")
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "# Curriculum Research: Test Topic" in content
    assert "mock-brain" in content
    assert "**Total Units:** 2" in content

    # Append Unit
    append_unit_to_markdown(
        topic="Test Topic",
        unit_index=1,
        total_units=2,
        unit_title="Introduction",
        passages=["This is a test passage."],
        facts=[{"statement": "The sky is blue", "confidence": 0.99}],
        sources=["https://example.com"],
    )

    content = file_path.read_text(encoding="utf-8")
    assert "## Unit 1/2: Introduction" in content
    assert "### Context & Narrative Summary" in content
    assert "This is a test passage." in content
    assert "### Distilled Atomic Facts" in content
    assert "**[Confidence: 0.99]** The sky is blue" in content
    assert "### Sources Consulted" in content
    assert "<https://example.com>" in content
