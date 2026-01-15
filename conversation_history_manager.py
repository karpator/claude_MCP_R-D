"""
Hierarchical Conversation History Manager
Manages conversation history by keeping recent messages in full detail
and progressively compressing older messages using Gemini.
"""

from typing import List, Dict, Optional
from google import genai
from google.genai import types


class ConversationHistoryManager:
    """
    Manages conversation history with hierarchical compression.

    Strategy:
    - Keep the last N messages verbatim (full detail)
    - Compress older messages into progressively condensed summaries
    - Use a separate Gemini model to perform compression
    """

    def __init__(
        self,
        project_id: str,
        region: str = "global",
        model: str = "gemini-2.0-flash-exp",
        recent_message_count: int = 5,
        compression_threshold: int = 10
    ):
        """
        Initialize the history manager.

        Args:
            project_id: Google Cloud project ID
            region: GCP region for Vertex AI
            model: Model to use for compression (lightweight model recommended)
            recent_message_count: Number of recent messages to keep verbatim
            compression_threshold: Trigger compression when history exceeds this length
        """
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=region
        )

        self.model = model
        self.recent_message_count = recent_message_count
        self.compression_threshold = compression_threshold

        # Store full conversation history
        self.full_history: List[types.Content] = []

        # Store compressed summaries of older conversations
        self.compressed_summaries: List[str] = []

    async def add_messages(self, contents: List[types.Content]):
        """
        Add multiple messages to the history at once and check if compression is needed.

        Args:
            contents: List of message contents (user and/or model)
        """
        # Add all messages
        self.full_history.extend(contents)
        #TODO: get rid of full_history, only keep recent + compressed summaries
        if len(self.full_history) > self.compression_threshold:
            uncompressed_count = len(self.full_history) - self.recent_message_count
            already_compressed = len(self.compressed_summaries)
            messages_to_compress = uncompressed_count - already_compressed

            if messages_to_compress > 0:
                await self._compress_older_messages(messages_to_compress)

    async def add_message(self, content: types.Content):
        """
        Add a single message to the history and check if compression is needed.

        Args:
            content: The message content (user or model)
        """
        await self.add_messages([content])

    def get_managed_history(self) -> List[types.Content]:
        """
        Get the managed history for sending to the main model.
        Compression already happened in add_message(), so just build the view.

        Returns:
            List of Content objects with recent messages + compressed summaries
        """
        # If history is below threshold, return as-is
        if len(self.full_history) <= self.compression_threshold:
            return self.full_history

        # Build managed history: compressed summary + recent messages
        managed_history = []

        # Add compressed summary as a system context if we have summaries
        if self.compressed_summaries:
            summary_text = self._build_summary_text()
            managed_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text=f"[Previous conversation summary: {summary_text}]"
                    )]
                )
            )

        # Add recent messages verbatim
        recent_messages = self.full_history[-self.recent_message_count:]
        managed_history.extend(recent_messages)

        return managed_history

    async def _compress_older_messages(self, count: int):
        """
        Compress a batch of older messages into a summary.

        Args:
            count: Number of messages to compress from the older part
        """
        # Get messages to compress (excluding recent ones)
        old_messages = self.full_history[:-self.recent_message_count]

        # Get the messages that haven't been compressed yet
        start_idx = len(self.compressed_summaries) * 2  # Each summary covers ~2 exchanges
        end_idx = start_idx + count
        messages_to_compress = old_messages[start_idx:end_idx]

        if not messages_to_compress:
            return

        # Build the conversation text for compression
        conversation_text = self._format_messages_for_compression(messages_to_compress)

        # Create compression prompt
        compression_prompt = f"""Summarize the following conversation exchange concisely, preserving key information, decisions, and context that might be relevant for future conversation turns. Focus on:
- Main topics discussed
- Important facts or data mentioned
- Decisions or conclusions reached
- User preferences or requirements stated

Conversation to summarize:
{conversation_text}

Provide a concise summary (2-3 sentences max):"""

        # Call Gemini to compress
        config = types.GenerateContentConfig(
            temperature=0.3,  # Lower temperature for consistent summaries
            max_output_tokens=200,
        )

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=compression_prompt,
            config=config,
        )

        summary = response.text.strip()
        self.compressed_summaries.append(summary)

    def _format_messages_for_compression(self, messages: List[types.Content]) -> str:
        """
        Format messages into readable text for compression.

        Args:
            messages: List of Content objects to format

        Returns:
            Formatted string representation
        """
        formatted = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"

            # Extract text from parts
            text_parts = []
            for part in msg.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

            if text_parts:
                formatted.append(f"{role}: {' '.join(text_parts)}")

        return "\n".join(formatted)

    def _build_summary_text(self) -> str:
        """
        Build a combined summary from all compressed summaries.

        Returns:
            Combined summary text
        """
        if len(self.compressed_summaries) == 1:
            return self.compressed_summaries[0]

        # For multiple summaries, combine them with context
        combined = []
        for i, summary in enumerate(self.compressed_summaries):
            combined.append(f"Part {i+1}: {summary}")

        return " | ".join(combined)

    def clear_history(self):
        """Clear all history and summaries."""
        self.full_history.clear()
        self.compressed_summaries.clear()

    def get_full_history(self) -> List[types.Content]:
        """
        Get the complete uncompressed history.

        Returns:
            Full conversation history
        """
        return self.full_history.copy()

    def get_history_stats(self) -> Dict[str, int]:
        """
        Get statistics about the current history state.

        Returns:
            Dictionary with history statistics
        """
        return {
            "total_messages": len(self.full_history),
            "recent_messages": min(self.recent_message_count, len(self.full_history)),
            "compressed_summaries": len(self.compressed_summaries),
            "compression_active": len(self.full_history) > self.compression_threshold
        }

