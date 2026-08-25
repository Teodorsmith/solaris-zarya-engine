"""
Tests for Differential Re-Learning:
- Differential vs baseline curriculum planning based on prior knowledge
- URL ingestion deduplication
- SHA-256 fact hash deduplication
- Cumulative markdown header enrichment
"""

import datetime
import unittest
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock

from agent.engine.exporter import get_topic_slug, init_markdown_note
from agent.engine.planner import CurriculumPlanner
from agent.memory.embeddings import EmbeddingEngine
from agent.memory.semantic import SemanticMemory
from agent.models import Fact


def _make_semantic() -> SemanticMemory:
    """Create an in-memory SemanticMemory with the fallback embedder for tests."""
    return SemanticMemory(":memory:", EmbeddingEngine(force_fallback=True))


def _make_fact(text: str, topic: str = "python") -> Fact:
    return Fact(
        id=0,
        text=text,
        confidence=0.9,
        source_type="web_ingestion",
        topic=topic,
        created_at=datetime.datetime.now(timezone.utc).isoformat(),
    )


class TestDifferentialPlanning(unittest.TestCase):
    def test_baseline_plan_when_no_prior_facts(self):
        """plan_curriculum uses foundational prompt when memory is empty."""
        brain = MagicMock()
        brain.generate.return_value = '["Unit 1: Basics", "Unit 2: Advanced"]'
        brain.extract_json.return_value = ["Unit 1: Basics", "Unit 2: Advanced"]

        semantic = _make_semantic()
        planner = CurriculumPlanner(brain, semantic=semantic)
        units = planner.plan_curriculum("python enums")

        self.assertEqual(units, ["Unit 1: Basics", "Unit 2: Advanced"])
        prompt_used = brain.generate.call_args[0][0]
        self.assertIn("deeply research and learn about", prompt_used)
        self.assertNotIn("enriching an existing knowledge base", prompt_used)

    def test_differential_plan_when_facts_exist(self):
        """plan_curriculum uses gap-filling prompt when prior facts exist."""
        brain = MagicMock()
        brain.generate.return_value = '["Advanced Enum Flags", "Enum Serialization"]'
        brain.extract_json.return_value = ["Advanced Enum Flags", "Enum Serialization"]

        semantic = _make_semantic()
        semantic.add_fact(_make_fact(
            "Python Enum is a class for creating enumerated constants.",
            topic="python enums",
        ))

        planner = CurriculumPlanner(brain, semantic=semantic)
        units = planner.plan_curriculum("python enums")

        prompt_used = brain.generate.call_args[0][0]
        self.assertIn("enriching an existing knowledge base", prompt_used)
        self.assertIn("differential study units", prompt_used)
        self.assertEqual(units, ["Advanced Enum Flags", "Enum Serialization"])

    def test_differential_plan_when_markdown_note_exists(self):
        """plan_curriculum detects prior units from an existing Markdown note."""
        brain = MagicMock()
        brain.generate.return_value = '["Gap Unit A"]'
        brain.extract_json.return_value = ["Gap Unit A"]

        semantic = _make_semantic()
        topic = "python enums differential md test"
        slug = get_topic_slug(topic)
        note_path = Path("data/knowledge") / f"{slug}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            "# Curriculum Research: python enums differential md test\n\n"
            "## Unit 1/2: Enum Basics\n\n"
            "## Unit 2/2: Enum Advanced\n\n",
            encoding="utf-8",
        )
        try:
            planner = CurriculumPlanner(brain, semantic=semantic)
            planner.plan_curriculum(topic)
            prompt_used = brain.generate.call_args[0][0]
            self.assertIn("Enum Basics", prompt_used)
            self.assertIn("enriching an existing knowledge base", prompt_used)
        finally:
            note_path.unlink(missing_ok=True)


class TestURLDeduplication(unittest.TestCase):
    def test_mark_and_check_url(self):
        """URLs are recognized as ingested after being marked."""
        semantic = _make_semantic()
        url = "https://docs.python.org/3/library/enum.html"
        self.assertFalse(semantic.is_url_ingested(url))
        semantic.mark_url_ingested(url)
        self.assertTrue(semantic.is_url_ingested(url))

    def test_url_canonicalization_fragment(self):
        """Fragments are stripped so the same page is not re-ingested."""
        semantic = _make_semantic()
        semantic.mark_url_ingested(
            "https://docs.python.org/3/library/enum.html#enum-members"
        )
        self.assertTrue(
            semantic.is_url_ingested("https://docs.python.org/3/library/enum.html")
        )

    def test_url_canonicalization_trailing_slash(self):
        """Trailing slashes are stripped for dedup consistency."""
        semantic = _make_semantic()
        semantic.mark_url_ingested("https://docs.python.org/3/library/enum.html/")
        self.assertTrue(
            semantic.is_url_ingested("https://docs.python.org/3/library/enum.html")
        )


class TestFactHashDeduplication(unittest.TestCase):
    def test_same_text_not_duplicated(self):
        """Inserting the same fact text twice returns False on second call."""
        semantic = _make_semantic()
        fact_text = "Python Enum members are instances of the Enum class."
        fact = _make_fact(fact_text)

        created1, fid1 = semantic.add_fact(fact)
        created2, fid2 = semantic.add_fact(_make_fact(fact_text))

        self.assertTrue(created1)
        self.assertFalse(created2)
        self.assertEqual(fid1, fid2)
        self.assertEqual(semantic.count(), 1)

    def test_different_texts_both_inserted(self):
        """Two distinct facts are both stored."""
        semantic = _make_semantic()
        semantic.add_fact(_make_fact("Enum members are unique by name."))
        semantic.add_fact(_make_fact("Enum values can be integers, strings, or tuples."))
        self.assertEqual(semantic.count(), 2)


class TestCumulativeMarkdown(unittest.TestCase):
    def setUp(self):
        self.topic = "python enums markdown test"
        self.slug = get_topic_slug(self.topic)
        self.note_path = Path("data/knowledge") / f"{self.slug}.md"
        self.note_path.parent.mkdir(parents=True, exist_ok=True)
        self.note_path.unlink(missing_ok=True)

    def tearDown(self):
        self.note_path.unlink(missing_ok=True)

    def test_init_creates_file(self):
        """init_markdown_note creates the file on first call."""
        init_markdown_note(self.topic, total_units=3, brain_model="mock")
        self.assertTrue(self.note_path.exists())
        content = self.note_path.read_text(encoding="utf-8")
        self.assertIn("Last Enriched:", content)
        self.assertIn("**Total Units:** 3", content)

    def test_enrichment_updates_header_not_content(self):
        """Calling init_markdown_note on existing file updates header, keeps unit content."""
        init_markdown_note(self.topic, total_units=3, brain_model="mock")
        with self.note_path.open("a", encoding="utf-8") as f:
            f.write("\n## Unit 1/3: Enum Basics\n\nSome content.\n\n---\n")

        init_markdown_note(self.topic, total_units=6, brain_model="mock")
        content = self.note_path.read_text(encoding="utf-8")

        self.assertIn("**Total Units:** 6", content)
        self.assertIn("## Unit 1/3: Enum Basics", content)
        self.assertIn("Some content.", content)


if __name__ == "__main__":
    unittest.main()
