"""
Queue-Based Conversation History Manager
Manages conversation history by keeping recent messages in a fixed-size queue
and maintaining a condensed summary of all older messages.
"""

from typing import List, Dict
from collections import deque
from google import genai
from google.genai import types


class ConversationHistoryManager:
    """
    Manages conversation history with queue-based compression.

    Strategy:
    - Keep the last N messages in a fixed-size queue
    - Maintain one condensed text summary of all older messages
    - When adding a new message: pop oldest from queue, compress it into summary, push new message
    - Return history as: condensed summary + recent queue messages
    """

    def __init__(
        self,
        project_id: str,
        region: str = "global",
        model: str = "gemini-2.0-flash-exp",
        queue_size: int = 3
    ):
        """
        Initialize the history manager.

        Args:
            project_id: Google Cloud project ID
            region: GCP region for Vertex AI
            model: Model to use for compression (lightweight model recommended)
            queue_size: Number of recent messages to keep in the queue
        """
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=region
        )

        self.model = model
        self.queue_size = queue_size

        # Fixed-size queue for recent messages
        self.message_queue: deque[types.Content] = deque(maxlen=queue_size)

        # Single condensed text summary of all older messages
        self.condensed_summary: str = ""


    async def add_messages(self, contents: List[types.Content]):
        """
        Add multiple messages to the history.
        For each message: if queue is full, pop oldest and compress it before adding new.

        Args:
            contents: List of message contents (user and/or model)
        """
        for content in contents:
            await self.add_message(content)

    async def add_message(self, content: types.Content):
        """
        Add a single message to the history.
        If queue is full, pop the oldest message, compress it into the summary, then add new message.

        Args:
            content: The message content (user or model)
        """
        # If queue is at capacity, compress the oldest message before adding new one
        if len(self.message_queue) == self.queue_size:
            oldest_message = self.message_queue[0]  # Peek at oldest without popping yet
            await self._compress_message_into_summary(oldest_message)

        # Add new message (deque will automatically pop oldest if at maxlen)
        self.message_queue.append(content)


    def get_managed_history(self) -> List[types.Content]:
        """
        Get the managed history for sending to the main model.
        Returns: condensed summary (if exists) + recent messages from queue

        Returns:
            List of Content objects with condensed summary + recent queue messages
        """
        managed_history = []

        # Add condensed summary as a user message if we have compressed history
        if self.condensed_summary:
            managed_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text=f"[Previous conversation summary: {self.condensed_summary}]"
                    )]
                )
            )

        # Add all messages from the queue in reverse order (oldest to newest)
        managed_history.extend(reversed(list(self.message_queue)))

        return managed_history


    async def _compress_message_into_summary(self, message: types.Content):
        """
        Compress a single message into the condensed summary.
        Takes the oldest message and merges it into the existing summary.

        Args:
            message: The message to compress into the summary
        """
        # Format the message
        message_text = self._format_message(message)

        # Create compression prompt
        if self.condensed_summary:
            compression_prompt = f"""You are managing a conversation history summary. You need to incorporate a new message into an existing summary.

Existing summary:
{self.condensed_summary}

New message to incorporate:
{message_text}

Provide an updated summary that includes the new message while maintaining key information from the existing summary. Keep it concise (2-4 sentences max):"""
        else:
            # First message to compress
            compression_prompt = f"""Summarize the following message concisely, preserving key information, decisions, and context that might be relevant for future conversation turns.

Message to summarize:
{message_text}

Provide a concise summary (6-7 sentences max):"""

        # Call Gemini to compress
        config = types.GenerateContentConfig(
            temperature=0.3,  # Lower temperature for consistent summaries
            max_output_tokens=300,
        )

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=compression_prompt,
            config=config,
        )

        self.condensed_summary = response.text.strip()

    def _format_message(self, message: types.Content) -> str:
        """
        Format a single message into readable text.

        Args:
            message: Content object to format

        Returns:
            Formatted string representation
        """
        role = "User" if message.role == "user" else "Assistant"

        # Extract text from parts
        text_parts = []
        for part in message.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)

        return f"{role}: {' '.join(text_parts)}" if text_parts else f"{role}: [no text content]"


    def clear_history(self):
        """Clear all history and summary."""
        self.message_queue.clear()
        self.condensed_summary = ""

    def get_queue_messages(self) -> List[types.Content]:
        """
        Get the current messages in the queue.

        Returns:
            List of messages in the queue
        """
        return list(self.message_queue)

    def get_condensed_summary(self) -> str:
        """
        Get the current condensed summary.

        Returns:
            The condensed summary text
        """
        return self.condensed_summary

    def get_history_stats(self) -> Dict[str, any]:
        """
        Get statistics about the current history state.

        Returns:
            Dictionary with history statistics
        """
        return {
            "queue_size": self.queue_size,
            "messages_in_queue": len(self.message_queue),
            "has_condensed_summary": bool(self.condensed_summary),
            "condensed_summary_length": len(self.condensed_summary) if self.condensed_summary else 0
        }

