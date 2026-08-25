import pytest
from unittest.mock import Mock, patch, MagicMock

import agent.cli as cli_mod
from agent.engine.chat import ChatEngine
from agent.models import EpisodicLog
from agent.cli import dispatch_command


@pytest.fixture
def mock_brain():
    brain = Mock()
    brain.generate.return_value = "Mock response"
    return brain


@pytest.fixture
def mock_episodic():
    ep = Mock()
    ep.conn = Mock()
    ep.log_event = Mock()
    return ep


@pytest.fixture
def mock_semantic():
    sem = Mock()
    sem.search.return_value = []
    return sem


@pytest.fixture
def mock_self_model():
    sm = Mock()
    sm._data = {"identity": "Test-Agent-v1"}
    return sm


def test_chat_engine_sliding_window(mock_brain, mock_episodic, mock_semantic, mock_self_model):
    # Simulate DB returning 20 chat messages (10 turns)
    mock_rows = []
    for i in range(20):
        kind = "chat_user" if i % 2 == 0 else "chat_assistant"
        mock_rows.append({"id": 20-i, "kind": kind, "content": f"msg {20-i}"})
    
    mock_episodic.conn.execute.return_value.fetchall.return_value = mock_rows
    
    # Mock row to log conversion
    def row_to_log(r):
        return EpisodicLog(kind=r["kind"], content=r["content"])
    mock_episodic._row_to_log.side_effect = row_to_log
    
    engine = ChatEngine(mock_brain, mock_episodic, mock_semantic, mock_self_model)
    engine.respond("hello")
    
    # Verify the prompt string format
    call_args = mock_brain.generate.call_args[0][0]
    
    assert "You are Test-Agent-v1" in call_args
    assert "Assistant: msg 1" in call_args
    assert "User: msg 20" in call_args
    assert "User: hello" in call_args
    
    # Verify persistence
    assert mock_episodic.log_event.call_count == 2
    args_list = mock_episodic.log_event.call_args_list
    assert args_list[0][0][0].kind == "chat_user"
    assert args_list[0][0][0].content == "hello"
    assert args_list[1][0][0].kind == "chat_assistant"
    assert args_list[1][0][0].content == "Mock response"


def test_chat_engine_clear_context_boundary(mock_brain, mock_episodic, mock_semantic, mock_self_model):
    # DB returns 5 messages, but middle one is chat_reset
    mock_rows = [
        {"id": 5, "kind": "chat_user", "content": "msg 5"},
        {"id": 4, "kind": "chat_assistant", "content": "msg 4"},
        {"id": 3, "kind": "chat_reset", "content": "reset"},
        {"id": 2, "kind": "chat_user", "content": "msg 2"},
        {"id": 1, "kind": "chat_assistant", "content": "msg 1"},
    ]
    mock_episodic.conn.execute.return_value.fetchall.return_value = mock_rows
    
    def row_to_log(r):
        return EpisodicLog(kind=r["kind"], content=r["content"])
    mock_episodic._row_to_log.side_effect = row_to_log
    
    engine = ChatEngine(mock_brain, mock_episodic, mock_semantic, mock_self_model)
    engine.respond("hello")
    
    call_args = mock_brain.generate.call_args[0][0]
    
    # Only msg 4 and 5 should be in history
    assert "msg 4" in call_args
    assert "msg 5" in call_args
    assert "msg 1" not in call_args
    assert "msg 2" not in call_args


def test_chat_engine_clear_method(mock_brain, mock_episodic, mock_semantic, mock_self_model):
    engine = ChatEngine(mock_brain, mock_episodic, mock_semantic, mock_self_model)
    engine.clear_context()
    
    args = mock_episodic.log_event.call_args[0][0]
    assert args.kind == "chat_reset"


def test_semantic_context_injection(mock_brain, mock_episodic, mock_semantic, mock_self_model):
    mock_episodic.conn.execute.return_value.fetchall.return_value = []
    
    fact_mock = Mock()
    fact_mock.text = "Project uses Python 3.10"
    mock_semantic.search.return_value = [fact_mock]
    
    engine = ChatEngine(mock_brain, mock_episodic, mock_semantic, mock_self_model)
    engine.respond("what python version?")
    
    call_args = mock_brain.generate.call_args[0][0]
    assert "[Project Context]" in call_args
    assert "Project uses Python 3.10" in call_args


@patch("agent.cli.console")
@patch("agent.cli.ChatEngine")
def test_unknown_command_routes_to_chat(mock_chat_cls, mock_console):
    # Setup mocks
    mock_chat_engine = Mock()
    mock_chat_cls.return_value = mock_chat_engine
    cli_mod.CHAT_FALLBACK_ENABLED = True
    
    dispatch_command(
        command="hello",
        rest="world",
        semantic=Mock(),
        episodic=Mock(),
        procedural=Mock(),
        project=Mock(),
        goals=Mock(),
        brain_or_manager=Mock(),
        self_model=Mock(),
    )
    
    # Verify fallback routing
    mock_chat_engine.respond.assert_called_with("hello world")


@patch("agent.cli.console")
@patch("agent.cli.ChatEngine")
def test_command_typo_does_not_route_to_chat(mock_chat_cls, mock_console):
    mock_chat_engine = Mock()
    mock_chat_cls.return_value = mock_chat_engine
    cli_mod.CHAT_FALLBACK_ENABLED = True
    
    dispatch_command(
        command="braiin",
        rest="switch gemini",
        semantic=Mock(),
        episodic=Mock(),
        procedural=Mock(),
        project=Mock(),
        goals=Mock(),
        brain_or_manager=Mock(),
        self_model=Mock(),
    )
    
    # Should catch typo and NOT call respond
    assert not mock_chat_engine.respond.called
    
    # Verify typo warning
    assert "Did you mean 'brain'?" in mock_console.print.call_args[0][0]
