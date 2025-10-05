# chat_memory.py
"""
Sliding-window conversation memory for a simple prompt-plus-response style chat.
"""

from collections import deque

class ChatMemory:
    def __init__(self, max_turns: int = 4):
        """
        max_turns: number of (user+bot) turns to keep. Each turn includes one user and one bot reply.
        """
        self.max_turns = max_turns
        # store as deque of (user_text, bot_text)
        self.history = deque(maxlen=self.max_turns)

    def add(self, user_text: str, bot_text: str):
        """Add a new turn to memory."""
        self.history.append((user_text.strip(), bot_text.strip()))

    def get_context(self) -> str:
        """
        Return the formatted context string to prepend to the next prompt.
        Format:
           User: <...>
           Bot: <...>
           ...
        """
        lines = []
        for user_text, bot_text in self.history:
            lines.append(f"User: {user_text}")
            lines.append(f"Bot: {bot_text}")
        if lines:
            return "\n".join(lines).rstrip() + "\n"
        else:
            return ""
