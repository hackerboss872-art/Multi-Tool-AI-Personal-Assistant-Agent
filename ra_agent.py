import asyncio

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


class ResearchAssistant:

    def __init__(self, mcp_tools):

        self.llm = ChatOllama(
            model="qwen3:0.6b",
            temperature=0.0,
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

        has_tool_results = any(
            isinstance(m, ToolMessage)
            for m in state.get("messages", [])
        )

        if not has_tool_results:
            system_prompt = f"""You are an expert, strictly factual Research Assistant.

Your objective is to gather real evidence before answering.

INSTRUCTIONS:
1. When asked about research topics, papers, weather, date/time, or calculations, you MUST invoke the appropriate tool first.
2. Available tools: {self.valid_tool_names}.
   - For academic papers or scientific research, invoke 'arxiv_search'.
   - For web information, recent news, or general topics, invoke 'web_search'.
   - For current weather in a city, invoke 'get_weather'.
   - For mathematical calculations, invoke 'calculator'.
   - For current date or time, invoke 'get_current_datetime'.
3. Do NOT generate the final research response or invent citations from memory. Invoke the tool first."""
        else:
            system_prompt = f"""You are an expert, strictly factual Research Assistant.

The tool execution is complete. Synthesize the tool findings into a concise, factual summary.
Do NOT call any more tools.

CRITICAL ANTI-HALLUCINATION RULES:
1. NEVER invent, fabricate, or guess author names, paper titles, publication dates, URLs, or arXiv IDs.
2. Every paper, source, or fact in your response MUST come directly from the tool output above.
3. Preserve the exact title, authors, date, and URL returned by the tool.
4. Do NOT claim that a source says something unless that information is directly in the tool output.
5. If any information cannot be verified, explicitly state: "Not verified from the available sources."

REQUIRED OUTPUT FORMAT:

Research Summary
- [Key factual points directly from the tool output]

Sources
1. [Exact title / location / service from tool] — [Exact URL returned by tool]

Verification
- [Clearly state what was verified from the tool output and what could not be verified]"""

        system_message = SystemMessage(
            content=system_prompt
        )

        model = self.llm if has_tool_results else self.llm_with_tools
        response = model.invoke(
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

                    if isinstance(observation, list):
                        text_parts = [
                            item.get("text", str(item)) if isinstance(item, dict) else str(item)
                            for item in observation
                        ]
                        output = "\n".join(text_parts)
                    elif isinstance(observation, dict) and "text" in observation:
                        output = str(observation["text"])
                    else:
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