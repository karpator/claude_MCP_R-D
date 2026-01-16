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
from google.genai.types import Modality, Tool, EnterpriseWebSearch, ThinkingConfig, ThinkingLevel, GoogleSearch, \
    ToolConfig, FunctionCallingConfig, FunctionCallingConfigMode
from jinja2 import Environment, FileSystemLoader
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from conversation_history_manager import ConversationHistoryManager


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

    async def connect_all(self):
        """Connect to all MCP servers with error handling."""
        connection_errors = []

        for name, url in self.endpoints.items():
            try:
                print(f"Connecting to {name} at {url}...")

                # Use AsyncExitStack to properly manage the context
                read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                    streamable_http_client(url)
                )

                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                await session.initialize()
                self.sessions[name] = session

                print(f"Connected to {name}")

            except asyncio.TimeoutError:
                error_msg = f"Timeout connecting to {name} at {url}"
                print(error_msg)
                connection_errors.append((name, "Timeout"))

            except ConnectionRefusedError:
                error_msg = f"Connection refused to {name} at {url} - Is the server running?"
                print(error_msg)
                connection_errors.append((name, "Connection refused"))

            except Exception as e:
                error_msg = f"Failed to connect to {name} at {url}: {type(e).__name__}: {e}"
                print(error_msg)
                connection_errors.append((name, str(e)))

        # Raise error if no servers connected successfully
        if not self.sessions:
            raise RuntimeError(
                f"Failed to connect to any MCP servers. Errors: {connection_errors}"
            )

        # Warn if some servers failed
        if connection_errors:
            print(f"\nWarning: {len(connection_errors)} server(s) failed to connect:")
            for name, error in connection_errors:
                print(f"  - {name}: {error}")

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
        self.model = "gemini-3-pro-preview"
        self.mcp_client: MultiServerClient = None
        self.enable_grounding = True
        self.force_tool_use = True

        # Initialize queue-based history manager
        self.history_manager = ConversationHistoryManager(
            project_id=project_id,
            region=region,
            model="gemini-2.5-flash",  # Lightweight model for compression
            queue_size=3  # Keep last 10 messages in queue
        )

    async def connect_servers(self, endpoints: Dict[str, str]):
        """
        Connect to multiple MCP servers.

        Args:
            endpoints: Dictionary of {server_name: url}
        """
        self.mcp_client = MultiServerClient(endpoints)
        await self.mcp_client.connect_all()
        await self.mcp_client.list_all_tools()

    async def send_message_stream(self, user_message: str, max_turns: int = 5):
        """
        Send a message to Gemini with streaming response and automatic tool calling.
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized. Call connect_servers first.")

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = Environment(loader=FileSystemLoader(current_dir), enable_async=True)
        system_prompt = await env.get_template("claude_MCP_R-D/templates/system_prompt.jinja2").render_async(
            language_code="hun",
        )

        managed_history = self.history_manager.get_managed_history()
        current_history = managed_history + [user_content]

        tools = [*self.mcp_client.get_sessions()]

        if self.enable_grounding:
            tools.append(Tool(google_search=GoogleSearch()))

        tool_config = None
        if self.force_tool_use:
            allowed_functions = []
            for session in self.mcp_client.sessions.values():
                tools_response = await session.list_tools()
                allowed_functions.extend([tool.name for tool in tools_response.tools])

            # Only configure function calling for MCP tools
            # Google Search will still be available but not forced
            if allowed_functions:
                tool_config = ToolConfig(
                    function_calling_config=FunctionCallingConfig(
                        mode=FunctionCallingConfigMode.ANY,
                        allowed_function_names=allowed_functions
                    )
                )

        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=8192,
            tools=tools,
            tool_config=tool_config,
            response_modalities=[Modality.TEXT],
            system_instruction=system_prompt,
            thinking_config=ThinkingConfig(include_thoughts=True, thinking_budget=1800)
        )

        response_message = await self.client.aio.models.generate_content(
            model=self.model,
            contents=current_history,
            config=generate_content_config,
        )

        # Merge user and assistant messages into a single Content object for history
        combined_content = types.Content(
            role="user",  # Store as user role to maintain conversation context
            parts=[
                types.Part.from_text(text=f"User: {user_message}"),
                types.Part.from_text(text=f"Assistant: {response_message.text}")
            ]
        )

        await self.history_manager.add_message(combined_content)

        yield response_message

    async def chat_loop(self):
        """Run an interactive chat loop."""
        print("\nGemini Multi-Server MCP Chat Started")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'tools' to see available tools")
        print("Type 'history' to see history statistics\n")

        while True:
            try:
                user_input = (await asyncio.to_thread(input, "You: ")).strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n Ending chat session...")
                    break

                if user_input.lower() == 'history':
                    stats = self.history_manager.get_history_stats()
                    print("\n History Statistics:")
                    print(f"  Total messages: {stats['total_messages']}")
                    print(f"  Recent messages (verbatim): {stats['recent_messages']}")
                    print(f"  Compressed summaries: {stats['compressed_summaries']}")
                    print(f"  Compression active: {stats['compression_active']}\n")
                    continue

                if user_input.lower() == 'tools':
                    await self.mcp_client.list_all_tools()
                    continue

                if not user_input:
                    continue

                # Get response from Gemini with streaming
                print("\nGemini: ", end="", flush=True)
                full_response = ""
                async for answer in self.send_message_stream(user_input):
                    print(answer.text, end="", flush=True)
                print("\n")  # New line after response

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
        "Elastic": os.getenv("MCP_SERVER_1", "http://localhost:8000/mcp"),
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


