"""
Gemini MCP Chat with Multi-Server Support
Connects to multiple HTTP MCP servers simultaneously using streamable HTTP client
Based on the MultiServerClient pattern
"""

import asyncio
import os
from typing import List, Dict
from contextlib import AsyncExitStack

from google import genai
from google.genai import types
from google.genai.types import Modality, Tool, EnterpriseWebSearch, ThinkingConfig, ThinkingLevel
from jinja2 import Environment, FileSystemLoader
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MultiServerClient:
    """
    Manages connections to multiple MCP servers via HTTP.
    Allows Gemini to use tools from multiple servers simultaneously.
    """

    def __init__(self, endpoints: Dict[str, str]):
        """
        Initialize with a dictionary of {server_name: sse_url}.

        Example:
            {
                "Calculator": "http://localhost:8000/mcp",
                "Weather": "http://localhost:8010/mcp"
            }

        Args:
            endpoints: Dictionary mapping server names to their HTTP URLs
        """
        self.endpoints = endpoints
        self.sessions = {}
        self._exit_stack = AsyncExitStack()
    #TODO: error elkapás
    async def connect_all(self):
        """Connect to all MCP servers."""
        for name, url in self.endpoints.items():
            print(f" Connecting to {name} at {url}...")

            # Use AsyncExitStack to properly manage the context
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(url)
            )

            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await session.initialize()
            self.sessions[name] = session

            print(f" Connected to {name}")

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        await self._exit_stack.aclose()

    async def list_all_tools(self):
        """List all tools from all connected servers."""
        for name, session in self.sessions.items():
            tools_response = await session.list_tools()
            print(f"\n Tools from {name}:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")

    async def call(self, server_name: str, tool_name: str, args: dict):
        """Call a tool on a specific server."""
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"No session for server '{server_name}'")
        return await session.call_tool(tool_name, args)

    def get_sessions(self):
        """Get all active MCP sessions for passing to Gemini."""
        return [session for session in self.sessions.values()]


class GeminiMCPChat:
    """
    Manages a chat session with Gemini using multiple MCP servers.
    """

    def __init__(self, project_id: str, region: str = "global"):
        """
        Initialize the chat session.

        Args:
            project_id: Google Cloud project ID for Vertex AI
            region: GCP region for Vertex AI (default: global)
        """
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=region
        )

        self.project_id = project_id
        self.region = region
        self.model = "gemini-2.5-flash"
        self.mcp_client: MultiServerClient = None
        self.conversation_history = []

    async def connect_servers(self, endpoints: Dict[str, str]):
        """
        Connect to multiple MCP servers.

        Args:
            endpoints: Dictionary of {server_name: url}
        """
        self.mcp_client = MultiServerClient(endpoints)
        await self.mcp_client.connect_all()
        await self.mcp_client.list_all_tools()

    async def send_message(self, user_message: str) -> str:
        """
        Send a message to Gemini with automatic MCP tool handling.

        Args:
            user_message: The user's message
            system_instruction: Optional system instruction

        Returns:
            Gemini's response text
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized. Call connect_servers first.")

        # Add user message to history
        self.conversation_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)]
            )
        )

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = Environment(loader=FileSystemLoader(current_dir), enable_async=True)

        system_prompt = await env.get_template("claude_MCP_R-D/templates/documents.jinja2").render_async(
            language_code="hun",
        )

        # Generate content with all MCP sessions as tools
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=8192,
            tools=[*self.mcp_client.get_sessions()],
            response_modalities=[Modality.TEXT],
            system_instruction=system_prompt,
            thinking_config=ThinkingConfig(include_thoughts=True,thinking_budget=1800)
        )

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=self.conversation_history,
            config=generate_content_config,
            #TODO: history két agentre (plusz egy history agent)
        )

        # Add response to history
        if response.candidates and response.candidates[0].content:
            self.conversation_history.append(response.candidates[0].content)

        return response.text
    #TODO: majd ezt használni
    async def send_message_stream(self, user_message: str, system_instruction: str = None):
        """
        Send a message to Gemini with streaming response.

        Args:
            user_message: The user's message
            system_instruction: Optional system instruction

        Yields:
            Response chunks from Gemini
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized. Call connect_servers first.")

        # Add user message to history
        self.conversation_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)]
            )
        )

        # Default system instruction
        if system_instruction is None:
            system_instruction = (
                "You are an AI assistant that helps the user in every way it can, "
                "without asking. Always use tools when necessary."
            )

        # Generate content with streaming
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=8192,
            tools=[*self.mcp_client.get_sessions()],
            response_modalities=[Modality.TEXT],
            system_instruction=system_instruction,
        )

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=self.conversation_history,
            config=generate_content_config,
        )

        async for chunk in response_stream:
            yield chunk

    async def chat_loop(self):
        """Run an interactive chat loop."""
        print("\n Gemini Multi-Server MCP Chat Started")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'tools' to see available tools\n")
        #TODO: add tool usage iteration
        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n Ending chat session...")
                    break

                if not user_input:
                    continue

                # Get response from Gemini
                print("\n Thinking...")
                response = await self.send_message(user_input)
                print(f"\n Gemini: {response}\n")

            except KeyboardInterrupt:
                print("\n\n Chat session interrupted...")
                break
            except Exception as e:
                print(f"\n Error: {e}\n")

    async def cleanup(self):
        """Clean up resources."""
        if self.mcp_client:
            await self.mcp_client.disconnect_all()


async def main():
    """
    Main function to run the Gemini MCP chat with multiple servers.
    """
    # Configuration
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "hu0092-bus-t-ai")
    REGION = os.getenv("VERTEX_REGION", "global")

    # Define MCP server endpoints
    # Add your MCP servers here
    endpoints = {
        "Calculator": os.getenv("MCP_SERVER_1", "http://localhost:8000/mcp"),
        # Add more servers as needed:
        # "Weather": "http://localhost:8010/mcp",
        # "Database": "http://localhost:8020/mcp",
    }

    print(" Starting Gemini Multi-Server MCP Chat...")
    print(f" Project: {PROJECT_ID}")
    print(f" Region: {REGION}")
    print(f" Servers: {', '.join(endpoints.keys())}\n")

    # Initialize chat session
    chat = GeminiMCPChat(project_id=PROJECT_ID, region=REGION)

    try:
        # Connect to all MCP servers
        await chat.connect_servers(endpoints)

        # Start chat loop
        await chat.chat_loop()

    finally:
        # Clean up
        await chat.cleanup()


if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())


