import asyncio

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


class ResearchAssistant:

    def __init__(self, mcp_tools):

        self.llm = ChatOllama(
            model="qwen3:0.6b",
            temperature=0.4,
            num_predict=1024
        )

        # Store MCP tools
        self.tools_list = mcp_tools

        # Create dictionary of tools by name
        self.tools_by_name = {
            tool.name: tool
            for tool in self.tools_list
        }

        # Make sure tools are available
        if not self.tools_list:
            raise ValueError("No tools provided")

        # Bind tools with LLM
        self.llm_with_tools = self.llm.bind_tools(
            self.tools_list
        )

        # Get valid tool names
        self.valid_tool_names = ", ".join(
            f"'{name}'"
            for name in self.tools_by_name
        )

    def call_llm(self, state):
        """Invoke the model with the research instructions and history."""

        system_message = SystemMessage(
            content=f"""
You are an expert Research Assistant.

Your goal is to research technical topics and generate
high-quality summaries that help users begin writing
their literature reviews.

RESEARCH PROCESS:

1. Identify the user's research question and scope.

2. Use available academic research tools to find papers.

3. Use web search tools to find relevant articles,
   documentation, and additional context.

4. Extract relevant content when extraction tools are available.

5. Compare research objectives, introductions, methodologies,
   findings, limitations, and conclusions.

6. Synthesize the evidence into a clear technical summary.

7. Include source URLs and bibliographic details returned
   by the tools whenever available.

8. Clearly state when evidence is insufficient.


TOOL RULES:

- Use only the tools bound to this agent.
- Available tool names: {self.valid_tool_names}
- Use the exact tool names provided by the MCP server.
- Supply arguments that match each tool's schema.
- Never invent tools, papers, findings, or citations.
- Use tool results as evidence for real-world factual claims.
- After receiving tool results, analyze them before answering.
- If you cannot find enough evidence, explain the limitation.


STRUCTURE THE FINAL ANSWER WHEN APPROPRIATE:

1. Research overview
2. Relevant papers and sources
3. Technical comparison
4. Key findings and research gaps
5. Conclusion and references
"""
        )

        response = self.llm_with_tools.invoke(
            [system_message] + state["messages"]
        )

        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    async def tool_node(self, state):
        """Execute the MCP tools requested by the model."""

        results = []

        # Check whether messages exist
        if not state.get("messages"):
            return {
                "messages": results
            }

        # Get the last message
        last_message = state["messages"][-1]

        # Get tool calls
        tool_calls = getattr(
            last_message,
            "tool_calls",
            []
        )

        # No tool calls
        if not tool_calls:
            return {
                "messages": results
            }

        # Execute every requested tool
        for tool_call in tool_calls:

            tool_name = tool_call["name"]

            tool_args = dict(
                tool_call.get("args") or {}
            )

            tool_call_id = tool_call["id"]

            # Find the tool
            tool = self.tools_by_name.get(
                tool_name
            )

            if tool is None:

                output = (
                    f"Unknown tool: {tool_name}"
                )

            else:

                try:

                    observation = await tool.ainvoke(
                        tool_args
                    )

                    output = str(observation)

                except Exception as e:

                    output = (
                        f"Tool execution failed: {e}"
                    )

            # Limit tool output size
            max_characters = 8000

            if len(output) > max_characters:

                output = (
                    f"Tool output truncated: "
                    f"{len(output)} characters > "
                    f"{max_characters}\n\n"
                    f"{output[:max_characters]}"
                )

            # Create ToolMessage
            results.append(
                ToolMessage(
                    content=output,
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
            )

        return {
            "messages": results
        }