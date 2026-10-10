import asyncio
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from services.ra_workflow_service import RAWorkflowService


async def main():

    print("=" * 60)
    print("Research Assistant")
    print("Model: Ollama Qwen3:0.6b")
    print("Transport: Local MCP Server (STDIO)")
    print("=" * 60)

    server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mcp_server.py"
    )

    client = MultiServerMCPClient(
        {
            "research_server": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [server_script]
            }
        }
    )

    print("\nConnecting to the MCP Server...")

    tools = await client.get_tools()

    if not tools:
        raise RuntimeError(
            "No tools returned from MCP server"
        )

    print(
        f"\nLoaded {len(tools)} tools from server"
    )

    for tool in tools:
        print(
            f"- {tool.name}: {tool.description}"
        )

    ra_service = RAWorkflowService(
        mcp_tools=tools
    )

    print("\nResearch Assistant is ready")
    print("Enter exit or quit to stop\n")

    while True:

        try:

            query = input("You: ").strip()

            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not query:
                continue

            print("\nResearching...\n")

            result = await ra_service.run_ra_workflow(
                query
            )

            messages = result.get(
                "messages",
                []
            )

            if messages:

                response = messages[-1].content

                print("-" * 60)
                print("Research Assistant")
                print("-" * 60)
                print(response)
                print("-" * 60)
                print()

            else:

                print("No response was generated.")

        except KeyboardInterrupt:

            print("\nExiting the agent...")
            break

        except Exception as error:

            print(f"\nError while executing: {error}\n")


if __name__ == "__main__":
    asyncio.run(main())