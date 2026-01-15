"""
Simple MCP Server Template
A minimal template for creating your own MCP server with custom tools

How to use:
1. Modify the TOOLS list to define your tool's interface
2. Implement your logic in the execute_tool() function
3. Run: python test_mcp_http_server.py
"""
import os
from datetime import datetime
import json
from typing import Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from jinja2 import FileSystemLoader, Environment
import vertexai
from vertexai.language_models import TextEmbeddingModel

from adapters.embedding_service import EmbeddingService
from common import RetriveNode
from post_retriver import AsyncPostRetriever

# Initialize Vertex AI and embedding model once at startup
vertexai.init()
embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

# ============================================================================
# TOOL DEFINITIONS - LOADED FROM JSON FILE
# ============================================================================

def load_tools(json_path: str = "mcp_tools.json") -> List[Dict[str, Any]]:
    """
    Load tool definitions from a JSON file.

    Args:
        json_path: Path to the JSON file containing tool definitions

    Returns:
        List of tool definitions
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, json_path)

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        print(f"Loaded {len(tools)} tool(s) from {json_path}")
        return tools
    except FileNotFoundError:
        print(f"Warning: {json_path} not found. Using empty tools list.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing {json_path}: {e}")
        return []

# Load tools at module initialization
TOOLS = load_tools()


# ============================================================================
# TOOL EXECUTION - IMPLEMENT YOUR LOGIC HERE
# ============================================================================

async def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> List[Dict[str, str]]:
    if tool_name == "retrive_documents":
        query = tool_args.get("query", "")
        keywords = tool_args.get("keywords", [])
        indices = tool_args.get("indices", [])


        embedder = EmbeddingService(embedding_model)
        query_embedding = embedder.get_embedding(query)
        retriver = RetriveNode()

        # Fix: Await the prep_async first, then pass result to exec_async
        prep_result = await retriver.prep_async(keywords, indices, query_embedding)
        documents = await retriver.exec_async(prep_result)
        delta_extraction = AsyncPostRetriever()
        documents = (await delta_extraction.exec_async(documents)).get("documents", [])
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = Environment(loader=FileSystemLoader(current_dir), enable_async=True)
        template = env.get_template("claude_MCP_R-D/templates/documents.jinja2")

        documents_str = await template.render_async(documents=documents)
        print(documents_str)
        result = {
            "documents": documents_str
        }

        return [{
            "type": "text",
            "text": json.dumps(result, ensure_ascii=False)
        }]

    return [{
        "type": "text",
        "text": f"Unknown tool: {tool_name}"
    }]


# ============================================================================
# FASTAPI SERVER - NO NEED TO MODIFY BELOW THIS LINE
# ============================================================================

app = FastAPI(title="ElasticSearch MCP Server", version="1.0.0")


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "name": "elastic-mcp-server",
        "version": "1.0.0",
        "description": "ES-MCP-Server",
        "endpoints": {
            "mcp": "/mcp",
            "tools": "/mcp/tools",
            "execute": "/mcp/execute"
        }
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP endpoint - JSON-RPC 2.0 protocol."""
    try:
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "elastic-mcp-server",
                        "version": "1.0.0"
                    }
                }
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": TOOLS
                }
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            result = await execute_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": result
                }
            }

        elif method == "notifications/initialized":
            return {}

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": body.get("id") if "body" in locals() else None,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }


@app.get("/mcp/tools")
async def list_tools():
    """List all available tools."""
    return {"tools": TOOLS}


@app.post("/mcp/execute")
async def execute_tool_endpoint(request: Request):
    """Execute a specific tool."""
    body = await request.json()
    tool_name = body.get("name")
    tool_args = body.get("arguments", {})

    if not tool_name:
        return JSONResponse(
            status_code=400,
            content={"error": "Tool name is required"}
        )

    result = await execute_tool(tool_name, tool_args)
    return {"content": result}


if __name__ == "__main__":
    print("Starting ElasticSearch MCP Server...")
    print(f"Loaded {len(TOOLS)} tool(s) from mcp_tools.json")
    print("Server will be available at: http://localhost:8000")
    print("MCP endpoint: http://localhost:8000/mcp")
    print("Tools endpoint: http://localhost:8000/mcp/tools")
    print("Press Ctrl+C to stop the server\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
