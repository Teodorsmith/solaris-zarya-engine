# Solaris Zarya Engine
# Copyright (C) 2026 Teodor Smith <teosmith.studios@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For commercial licensing options without AGPLv3 network-copyleft obligations,
# contact: teosmith.studios@gmail.com

from rich.console import Console
from rich.markdown import Markdown

from agent.brains.base import BaseBrain
from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.self_model import SelfModel
from agent.models import EpisodicLog

console = Console()

class ChatEngine:
    """Conversational fallback engine maintaining a multi-turn context window."""

    def __init__(
        self,
        brain: BaseBrain,
        episodic_mem: EpisodicMemory,
        semantic_mem: SemanticMemory,
        self_model: SelfModel,
    ):
        self.brain = brain
        self.episodic = episodic_mem
        self.semantic = semantic_mem
        self.self_model = self_model

    def respond(self, user_input: str) -> None:
        """Process conversational input, stream/print response, and save to history."""
        # 1. Fetch context window
        rows = self.episodic.conn.execute(
            "SELECT * FROM episodic_log "
            "WHERE kind IN ('chat_user', 'chat_assistant', 'chat_reset') "
            "ORDER BY id DESC LIMIT 20"
        ).fetchall()

        logs = [self.episodic._row_to_log(r) for r in rows]

        history = []
        for log in logs:
            if log.kind == "chat_reset":
                break
            history.append(log)

        history.reverse()

        # 2. Inject Persona
        persona = self.self_model._data.get("identity", "Autonomous-Agent-v1")
        system_prompt = (
            f"You are {persona}.\n"
            "You are helpful, honest, and grounded.\n"
            "You may use project context when relevant.\n"
        )

        # 3. Inject Semantic Context
        # Using a fast semantic lookup to find grounded context
        facts = self.semantic.search(user_input, top_k=2)
        project_context = ""
        if facts:
            project_context = "[Project Context]\n" + "\n".join(
                f"- {f.text}" for f in facts
            ) + "\n"

        # 4. Construct Prompt
        prompt = system_prompt
        if project_context:
            prompt += f"\n{project_context}\n"

        prompt += "\n[Conversation History]\n"
        for log in history:
            if log.kind == "chat_user":
                prompt += f"User: {log.content}\n"
            elif log.kind == "chat_assistant":
                prompt += f"Assistant: {log.content}\n"

        prompt += f"User: {user_input}\nAssistant:"

        # 5. Generate & Print
        response = self.brain.generate(prompt)
        console.print(Markdown(response))

        # 6. Persist
        self.episodic.log_event(
            EpisodicLog(kind="chat_user", content=user_input)
        )
        self.episodic.log_event(
            EpisodicLog(kind="chat_assistant", content=response)
        )

    def clear_context(self) -> None:
        """Reset the conversation context by appending a boundary event."""
        self.episodic.log_event(
            EpisodicLog(
                kind="chat_reset", 
                content="User explicitly cleared chat history."
            )
        )
        console.print("[green]Chat context cleared.[/green]")
