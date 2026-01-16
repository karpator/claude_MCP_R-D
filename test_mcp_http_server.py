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
import httpx
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
        #print(documents_str)
        result = {
            "documents": documents_str
        }

        return [{
            "type": "text",
            "text": json.dumps(result, ensure_ascii=False)
        }]

    elif tool_name == "get_weather":
        city = tool_args.get("city")
        country_code = tool_args.get("country_code", "")
        units = tool_args.get("units", "metric")

        try:
            # Using Open-Meteo API (free, no API key required)
            # First, get coordinates for the city using geocoding
            location_query = f"{city},{country_code}" if country_code else city

            async with httpx.AsyncClient() as client:
                # Geocoding API to get coordinates
                geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search"
                geo_params = {
                    "name": city,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                }

                geo_response = await client.get(geocoding_url, params=geo_params)
                geo_data = geo_response.json()

                if not geo_data.get("results"):
                    return [{
                        "type": "text",
                        "text": json.dumps({
                            "error": f"City '{city}' not found. Please check the spelling or try adding a country code."
                        }, ensure_ascii=False)
                    }]

                location = geo_data["results"][0]
                latitude = location["latitude"]
                longitude = location["longitude"]
                location_name = location.get("name", city)
                country = location.get("country", "")

                # Weather API to get current weather
                weather_url = "https://api.open-meteo.com/v1/forecast"
                weather_params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m",
                    "temperature_unit": "celsius" if units == "metric" else "fahrenheit",
                    "wind_speed_unit": "kmh" if units == "metric" else "mph",
                    "timezone": "auto"
                }

                weather_response = await client.get(weather_url, params=weather_params)
                weather_data = weather_response.json()

                current = weather_data.get("current", {})

                # Weather code interpretation
                weather_code = current.get("weather_code", 0)
                weather_descriptions = {
                    0: "Clear sky",
                    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                    45: "Foggy", 48: "Depositing rime fog",
                    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
                    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
                    77: "Snow grains",
                    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
                    85: "Slight snow showers", 86: "Heavy snow showers",
                    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
                }
                weather_condition = weather_descriptions.get(weather_code, "Unknown")

                temp_unit = "°C" if units == "metric" else "°F"
                wind_unit = "km/h" if units == "metric" else "mph"

                result = {
                    "location": f"{location_name}, {country}",
                    "coordinates": f"{latitude}, {longitude}",
                    "temperature": f"{current.get('temperature_2m', 'N/A')}{temp_unit}",
                    "feels_like": f"{current.get('apparent_temperature', 'N/A')}{temp_unit}",
                    "weather": weather_condition,
                    "humidity": f"{current.get('relative_humidity_2m', 'N/A')}%",
                    "wind_speed": f"{current.get('wind_speed_10m', 'N/A')} {wind_unit}",
                    "wind_direction": f"{current.get('wind_direction_10m', 'N/A')}°",
                    "cloud_cover": f"{current.get('cloud_cover', 'N/A')}%",
                    "precipitation": f"{current.get('precipitation', 'N/A')} mm",
                    "timestamp": current.get('time', 'N/A')
                }

                return [{
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2)
                }]

        except httpx.HTTPError as e:
            return [{
                "type": "text",
                "text": json.dumps({
                    "error": f"Weather API request failed: {str(e)}"
                }, ensure_ascii=False)
            }]
        except Exception as e:
            return [{
                "type": "text",
                "text": json.dumps({
                    "error": f"Failed to get weather data: {str(e)}"
                }, ensure_ascii=False)
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
