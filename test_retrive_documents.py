"""
Test script for the retrive_documents MCP tool
Tests the ability to retrieve documents from ElasticSearch via the MCP server
"""
import asyncio
import json
import httpx


async def test_retrive_documents():
    """Test the retrive_documents tool with pipeline_solaw_test index"""

    # MCP server URL
    mcp_url = "http://localhost:8000/mcp"

    # Test query parameters
    test_query = {
        "query": "What are the main topics in these documents?",
        "keywords": ["document", "information"],
        "indices": ["pipeline_solaw_test"]
    }

    print("🧪 Testing retrive_documents tool")
    print(f"📊 Query: {test_query['query']}")
    print(f"🔑 Keywords: {test_query['keywords']}")
    print(f"📚 Indices: {test_query['indices']}")
    print()

    # Create the MCP JSON-RPC request
    request_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "retrive_documents",
            "arguments": test_query
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("🔄 Sending request to MCP server...")
            response = await client.post(mcp_url, json=request_payload)

            if response.status_code == 200:
                result = response.json()
                print("✅ Request successful!")
                print()

                if "result" in result:
                    content = result["result"]["content"]
                    print(f"📦 Response content type: {type(content)}")
                    print(f"📦 Number of content items: {len(content)}")
                    print()

                    if content and len(content) > 0:
                        first_item = content[0]
                        print(f"🔍 First content item type: {first_item.get('type')}")

                        if first_item.get('type') == 'text':
                            text_content = first_item.get('text', '')

                            # Try to parse as JSON
                            try:
                                parsed_data = json.loads(text_content)
                                print("✅ Response is valid JSON")
                                print()

                                if 'documents' in parsed_data:
                                    documents = parsed_data['documents']
                                    print(f"📄 Documents retrieved (length): {len(documents) if isinstance(documents, str) else 'N/A'}")
                                    print()
                                    print("📝 Documents content preview:")
                                    print("-" * 80)
                                    if isinstance(documents, str):
                                        # Preview first 500 characters
                                        preview = documents[:500]
                                        print(preview)
                                        if len(documents) > 500:
                                            print(f"\n... (truncated, total length: {len(documents)} characters)")
                                    else:
                                        print(documents)
                                    print("-" * 80)
                                else:
                                    print("⚠️ No 'documents' key in response")
                                    print(f"Response keys: {list(parsed_data.keys())}")

                            except json.JSONDecodeError as e:
                                print(f"⚠️ Response is not valid JSON: {e}")
                                print(f"Raw text preview: {text_content[:200]}...")
                        else:
                            print(f"⚠️ Unexpected content type: {first_item.get('type')}")
                    else:
                        print("⚠️ No content in response")

                elif "error" in result:
                    print(f"❌ MCP Error: {result['error']}")
                else:
                    print(f"⚠️ Unexpected response format: {result}")

            else:
                print(f"❌ HTTP Error {response.status_code}")
                print(f"Response: {response.text}")

    except httpx.ConnectError:
        print("❌ Failed to connect to MCP server at http://localhost:8000")
        print("Make sure the server is running with: python test_mcp_http_server.py")

    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


async def test_list_tools():
    """Test that the tools list endpoint works"""
    print("\n🧪 Testing tools/list endpoint")
    print()

    mcp_url = "http://localhost:8000/mcp"
    request_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(mcp_url, json=request_payload)

            if response.status_code == 200:
                result = response.json()
                if "result" in result and "tools" in result["result"]:
                    tools = result["result"]["tools"]
                    print(f"✅ Found {len(tools)} tool(s):")
                    for tool in tools:
                        print(f"  • {tool['name']}: {tool['description']}")
                else:
                    print(f"⚠️ Unexpected response: {result}")
            else:
                print(f"❌ HTTP Error {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all tests"""
    print("=" * 80)
    print("🚀 MCP Server Test Suite")
    print("=" * 80)
    print()

    # First check if tools list works
    await test_list_tools()

    print()
    print("=" * 80)

    # Then test the actual document retrieval
    await test_retrive_documents()

    print()
    print("=" * 80)
    print("✨ Test suite completed")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

