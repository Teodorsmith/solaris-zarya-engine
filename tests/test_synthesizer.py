import pytest
from unittest.mock import Mock, patch, MagicMock
from agent.engine.synthesizer import (
    SkillSynthesizer,
    KnowledgeSynthesizer,
    SynthesizerError,
)
from agent.brains.mock_brain import MockBrain
from agent.models import Fact


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------

def test_knowledge_synthesizer_distill():
    brain = MockBrain()
    semantic = Mock()
    fact_obj = Fact(
        id=1,
        text="Python is a programming language.",
        confidence=0.8,
        source_type="web_ingestion",
        topic="python",
        created_at="",
    )
    semantic.add_fact.return_value = (True, fact_obj)
    synth = KnowledgeSynthesizer(brain, semantic)

    passage_text = (
        "Python was created by Guido van Rossum and first released in 1991. "
        "Its design philosophy emphasizes code readability with its notable "
        "use of significant indentation. Its language constructs and object "
        "oriented approach aim to help programmers write clear, logical code "
        "for small and large-scale projects."
    )
    with patch.object(
        brain,
        'generate',
        side_effect=[
            '["Python is a programming language."]',
            f'["{passage_text}"]',
        ],
    ):
        facts, passages = synth.distill_to_semantic_db(
            "Some text about python", "python"
        )
        assert len(facts) == 1
        assert facts[0].text == "Python is a programming language."
        assert len(passages) == 1
        semantic.add_fact.assert_called_once()
        semantic.add_passage.assert_called_once()


def test_knowledge_synthesizer_distill_invalid_json():
    brain = MockBrain()
    semantic = Mock()
    synth = KnowledgeSynthesizer(brain, semantic)

    with patch.object(brain, 'generate', return_value='Not a JSON'):
        facts, passages = synth.distill_to_semantic_db(
            "Some text about python", "python"
        )
        assert len(facts) == 0
        assert len(passages) == 0


def test_skill_synthesizer_learn_skill_success():
    brain = MockBrain()
    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    validator.validate_and_run.return_value = {}
    governor = Mock()
    governor.request_skill_write_permission.return_value = True

    synth = SkillSynthesizer(brain, retriever, procedural, validator, governor)

    with patch.object(
        brain,
        'generate',
        return_value='''
{
  "skill_name": "test_skill",
  "description": "Test description",
  "code": "def execute(): pass",
  "test_code": "def test_execute(): pass"
}
''',
    ):
        from unittest.mock import mock_open

        with patch('builtins.open', mock_open()):
            skill = synth.learn_skill("Test topic")
            assert skill is not None
            assert skill.name == "test_skill"
            procedural.register.assert_called_once()


def test_skill_synthesizer_validation_failure_retries():
    brain = MockBrain()
    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    governor = Mock()
    governor.request_skill_write_permission.return_value = True

    from agent.engine.validator import SecurityError

    validator.validate_and_run.side_effect = [
        SecurityError("Failed test_execute"),
        {},
    ]

    synth = SkillSynthesizer(brain, retriever, procedural, validator, governor)

    with patch.object(
        brain,
        'generate',
        return_value='''
{
  "skill_name": "test_skill",
  "description": "Test description",
  "code": "def execute(): pass",
  "test_code": "def test_execute(): pass"
}
''',
    ):
        from unittest.mock import mock_open

        with patch('builtins.open', mock_open()):
            with patch('time.sleep'):
                skill = synth.learn_skill("Test topic")
                assert skill is not None
                assert skill.name == "test_skill"
                assert validator.validate_and_run.call_count == 2
                procedural.register.assert_called_once()


def test_skill_synthesizer_exhausts_retries():
    brain = MockBrain()
    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    governor = Mock()

    from agent.engine.validator import SecurityError

    validator.validate_and_run.side_effect = SecurityError("Always fails")

    synth = SkillSynthesizer(brain, retriever, procedural, validator, governor)

    with patch.object(
        brain,
        'generate',
        return_value='''
{
  "skill_name": "test_skill",
  "description": "Test description",
  "code": "def execute(): pass",
  "test_code": "def test_execute(): pass"
}
''',
    ):
        with patch('time.sleep'):
            with pytest.raises(
                SynthesizerError, match="Failed to synthesize skill"
            ):
                synth.learn_skill("Test topic")


def test_skill_synthesizer_bad_json():
    brain = MockBrain()
    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    governor = Mock()

    synth = SkillSynthesizer(brain, retriever, procedural, validator, governor)

    with patch.object(brain, 'generate', return_value='Invalid JSON'):
        with patch('time.sleep'):
            with pytest.raises(
                SynthesizerError, match="Failed to synthesize skill"
            ):
                synth.learn_skill("Test topic")


def test_skill_synthesizer_records_reasoning_episode():
    brain = MockBrain()
    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    validator.validate_and_run.return_value = {}
    governor = Mock()
    governor.request_skill_write_permission.return_value = True
    reasoning_mem = Mock()

    synth = SkillSynthesizer(
        brain,
        retriever,
        procedural,
        validator,
        governor,
        reasoning_memory=reasoning_mem,
    )

    with patch.object(
        brain,
        'generate',
        return_value='''
{
  "skill_name": "recorded_skill",
  "description": "Test description",
  "code": "def execute(**kwargs): return {'result': 42}",
  "test_code": "def test_execute(): pass"
}
''',
    ):
        from unittest.mock import mock_open

        with patch('builtins.open', mock_open()):
            skill = synth.learn_skill("Recorded topic")
            assert skill is not None
            assert skill.name == "recorded_skill"
            assert reasoning_mem.log_episode.called
            episode = reasoning_mem.log_episode.call_args[0][0]
            assert episode.outcome_class == "success"
            assert episode.verified is True
            assert "recorded_skill" in episode.action


# ---------------------------------------------------------------------------
# NEW: DPO harvesting, MockBrain guard, episodic logging, redaction
# ---------------------------------------------------------------------------

_GOOD_RESPONSE = '''
{
  "skill_name": "repair_skill",
  "description": "A skill that needed repair",
  "code": "def execute(**kwargs): return {'result': 1}",
  "test_code": "def test_execute(): pass"
}
'''


def test_repair_triggers_dataset_harvest():
    """A multi-attempt synthesis that repairs should call harvest_dpo_pair exactly once."""
    from agent.engine.validator import SecurityError

    # Use a non-MockBrain so the guard allows harvesting
    brain = Mock()
    brain.generate.return_value = _GOOD_RESPONSE
    brain.extract_json.side_effect = lambda r: {
        "skill_name": "repair_skill",
        "description": "A skill that needed repair",
        "code": "def execute(**kwargs): return {'result': 1}",
        "test_code": "def test_execute(): pass",
    }

    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    validator.validate_and_run.side_effect = [SecurityError("first fail"), {}]
    governor = Mock()
    governor.request_skill_write_permission.return_value = True

    dataset_builder = Mock()
    dataset_builder.harvest_dpo_pair.return_value = True

    synth = SkillSynthesizer(
        brain, retriever, procedural, validator, governor,
        dataset_builder=dataset_builder,
    )

    with patch('builtins.open', MagicMock()):
        with patch('time.sleep'):
            with patch('os.makedirs'):
                skill = synth.learn_skill("repair topic")

    assert skill is not None
    dataset_builder.harvest_dpo_pair.assert_called_once()
    call_kwargs = dataset_builder.harvest_dpo_pair.call_args
    assert call_kwargs.kwargs["prompt"] == "repair topic"
    assert call_kwargs.kwargs["metadata"]["source"] == "skill_synthesizer"
    assert call_kwargs.kwargs["metadata"]["success_source"] == "mock_only"
    assert call_kwargs.kwargs["metadata"]["chosen_exit_code"] == 0
    assert call_kwargs.kwargs["metadata"]["rejected_exit_code"] == 1


def test_mockbrain_does_not_harvest():
    """When the active brain is MockBrain, no DPO pair should be harvested."""
    from agent.engine.validator import SecurityError

    brain = MockBrain()  # explicitly MockBrain
    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    validator.validate_and_run.side_effect = [SecurityError("fail once"), {}]
    governor = Mock()
    governor.request_skill_write_permission.return_value = True

    dataset_builder = Mock()
    dataset_builder.harvest_dpo_pair.return_value = True

    synth = SkillSynthesizer(
        brain, retriever, procedural, validator, governor,
        dataset_builder=dataset_builder,
    )

    with patch.object(brain, 'generate', return_value=_GOOD_RESPONSE):
        with patch('builtins.open', MagicMock()):
            with patch('time.sleep'):
                with patch('os.makedirs'):
                    skill = synth.learn_skill("mock topic")

    assert skill is not None
    # Must NOT harvest when using MockBrain
    dataset_builder.harvest_dpo_pair.assert_not_called()


def test_episodic_skill_repair_resolved_logged():
    """On successful repair, skill_repair_resolved must be logged to episodic memory."""
    from agent.engine.validator import SecurityError

    brain = Mock()
    brain.generate.return_value = _GOOD_RESPONSE
    brain.extract_json.return_value = {
        "skill_name": "repair_skill",
        "description": "A skill that needed repair",
        "code": "def execute(**kwargs): return {'result': 1}",
        "test_code": "def test_execute(): pass",
    }

    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    validator.validate_and_run.side_effect = [SecurityError("fail once"), {}]
    governor = Mock()
    governor.request_skill_write_permission.return_value = True
    episodic = Mock()

    synth = SkillSynthesizer(
        brain, retriever, procedural, validator, governor,
        episodic_memory=episodic,
    )

    with patch('builtins.open', MagicMock()):
        with patch('time.sleep'):
            with patch('os.makedirs'):
                skill = synth.learn_skill("repair topic")

    assert skill is not None
    assert episodic.log_event.called
    logged = episodic.log_event.call_args[0][0]
    assert logged.kind == "skill_repair_resolved"
    assert logged.outcome == "success"


def test_secret_redaction():
    """_redact_secrets should mask all known secret patterns."""
    cases = [
        # OpenAI-style token — check full token body is gone
        ("sk-abcdef1234567890abcdef", "abcdef1234567890abcdef"),
        # Generic api_key assignment
        ("api_key = 'supersecret123'", "supersecret123"),
        # AWS AKIA key
        ("AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
        # Bearer token body
        ("Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.abc123def456", "eyJhbGciOiJSUzI1NiJ9"),
        # GitHub PAT (exact length required)
        ("ghp_" + "A" * 36, "ghp_" + "A" * 36),
    ]

    for secret, substring in cases:
        code = f"x = '{secret}'\nother = 42"
        redacted = SkillSynthesizer._redact_secrets(code)
        assert substring not in redacted, (
            f"Expected '{substring}' to be redacted but it survived in: {redacted!r}"
        )
        assert "other = 42" in redacted  # non-secret lines must survive


def test_first_attempt_success_no_harvest():
    """When the first attempt succeeds there is no failed candidate, so harvest_dpo_pair must NOT be called."""
    brain = Mock()
    brain.generate.return_value = _GOOD_RESPONSE
    brain.extract_json.return_value = {
        "skill_name": "repair_skill",
        "description": "desc",
        "code": "def execute(**kwargs): return {'result': 1}",
        "test_code": "def test_execute(): pass",
    }
    retriever = Mock()
    retriever.retrieve.return_value = Mock(facts=[])
    procedural = Mock()
    validator = Mock()
    validator.validate_and_run.return_value = {}  # first attempt passes
    governor = Mock()
    governor.request_skill_write_permission.return_value = True
    dataset_builder = Mock()

    synth = SkillSynthesizer(
        brain, retriever, procedural, validator, governor,
        dataset_builder=dataset_builder,
    )

    with patch('builtins.open', MagicMock()):
        with patch('os.makedirs'):
            skill = synth.learn_skill("clean topic")

    assert skill is not None
    dataset_builder.harvest_dpo_pair.assert_not_called()
