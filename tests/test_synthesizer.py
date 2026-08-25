import pytest
from unittest.mock import Mock, patch
from agent.engine.synthesizer import SkillSynthesizer, KnowledgeSynthesizer, SynthesizerError
from agent.brains.mock_brain import MockBrain
from agent.models import Fact

def test_knowledge_synthesizer_distill():
    brain = MockBrain()
    semantic = Mock()
    # add_fact returns (created, returned_fact)
    semantic.add_fact.return_value = (True, Fact(id=1, text="Python is a programming language.", confidence=0.8, source_type="web_ingestion", topic="python", created_at=""))
    synth = KnowledgeSynthesizer(brain, semantic)
    
    # Needs to return a list of strings
    with patch.object(brain, 'generate', side_effect=[
        '["Python is a programming language."]', # facts response
        '["Python was created by Guido van Rossum and first released in 1991. Its design philosophy emphasizes code readability with its notable use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects."]' # passages response
    ]):
        facts, passages = synth.distill_to_semantic_db("Some text about python", "python")
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
        facts, passages = synth.distill_to_semantic_db("Some text about python", "python")
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
    
    with patch.object(brain, 'generate', return_value='''
{
  "skill_name": "test_skill",
  "description": "Test description",
  "code": "def execute(): pass",
  "test_code": "def test_execute(): pass"
}
'''):
        # Mock file writing
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
    
    # First validation fails with SecurityError, second succeeds
    validator.validate_and_run.side_effect = [
        SecurityError("Failed test_execute"),
        {}
    ]
    
    synth = SkillSynthesizer(brain, retriever, procedural, validator, governor)
    
    with patch.object(brain, 'generate', return_value='''
{
  "skill_name": "test_skill",
  "description": "Test description",
  "code": "def execute(): pass",
  "test_code": "def test_execute(): pass"
}
'''):
        from unittest.mock import mock_open
        with patch('builtins.open', mock_open()):
            # Also patch sleep so we don't wait
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
    
    # Always fails
    validator.validate_and_run.side_effect = SecurityError("Always fails")
    
    synth = SkillSynthesizer(brain, retriever, procedural, validator, governor)
    
    with patch.object(brain, 'generate', return_value='''
{
  "skill_name": "test_skill",
  "description": "Test description",
  "code": "def execute(): pass",
  "test_code": "def test_execute(): pass"
}
'''):
        with patch('time.sleep'):
            with pytest.raises(SynthesizerError, match="Failed to synthesize skill"):
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
            with pytest.raises(SynthesizerError, match="Failed to synthesize skill"):
                synth.learn_skill("Test topic")
