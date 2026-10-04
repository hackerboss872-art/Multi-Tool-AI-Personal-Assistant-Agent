"""
Local AI Agent using LangChain + Ollama (Qwen 2.5 3B)

Tools:
1. Calculator
2. Web Search
3. Job Recommendation
4. Resume RAG
"""

import sys

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from tools import tools as external_tools
from system_prompt import SYSTEM_PROMPT


# ==========================================
# 1. CALCULATOR TOOL
# ==========================================

@tool
def calculator(expression: str) -> str:
    """
    Evaluate basic mathematical calculations.

    Examples:
    25 * 48
    15% of 800
    100 / 4
    """

    try:
        expr = (
            str(expression)
            .lower()
            .replace("of", "*")
            .replace("%", "/100")
        )

        allowed = set("0123456789+-*/(). ")
        sanitized = "".join(
            char for char in expr
            if char in allowed
        )

        if not sanitized.strip():
            return "Error: Invalid mathematical expression."

        result = eval(
            sanitized,
            {"__builtins__": {}},
            {}
        )

        return f"Calculation Result: {result}"

    except Exception as err:
        return f"Calculator Error: {str(err)}"


# ==========================================
# 2. ALL AGENT TOOLS
# ==========================================

# tools.py already contains:
# - web_search
# - get_job_recommendation
# - get_resume_data

tools = [
    calculator,
    *external_tools
]

tools_by_name = {
    tool_item.name: tool_item
    for tool_item in tools
}


# ==========================================
# 3. LLM SETUP
# ==========================================

def get_agent_model():

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0.3
    )

    return llm.bind_tools(tools)


# ==========================================
# 4. AGENT PROCESSING
# ==========================================

def process_user_query(llm_with_tools, user_input: str) -> str:

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]

    # Allow multiple tool calls if necessary
    for _ in range(5):

        ai_msg = llm_with_tools.invoke(messages)

        # No tool required
        if not getattr(ai_msg, "tool_calls", None):
            return ai_msg.content

        # Add AI message containing tool calls
        messages.append(ai_msg)

        # Execute requested tools
        for tool_call in ai_msg.tool_calls:

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")

            selected_tool = tools_by_name.get(tool_name)

            if selected_tool:

                try:
                    tool_result = selected_tool.invoke(tool_args)

                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id
                        )
                    )

                except Exception as err:

                    messages.append(
                        ToolMessage(
                            content=f"Tool Error: {err}",
                            tool_call_id=tool_id
                        )
                    )

            else:

                messages.append(
                    ToolMessage(
                        content=f"Error: Tool '{tool_name}' not found.",
                        tool_call_id=tool_id
                    )
                )

    return "I could not complete the request after several tool calls."


# ==========================================
# 5. TERMINAL CHAT
# ==========================================

def main():

    print("=" * 60)
    print("LOCAL AI AGENT")
    print("=" * 60)
    print("Model: Qwen 2.5 3B")
    print("LLM: Ollama")
    print()
    print("Available tools:")
    print("- Calculator")
    print("- Web Search")
    print("- Job Recommendation")
    print("- Resume RAG")
    print()
    print("Type 'exit' to quit.")
    print("=" * 60)

    try:

        llm_with_tools = get_agent_model()

    except Exception as err:

        print(f"Error initializing Ollama model: {err}")
        sys.exit(1)

    while True:

        try:

            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:

                print("Goodbye!")
                break

            response = process_user_query(
                llm_with_tools,
                user_input
            )

            print(f"\nAI: {response.strip()}")

        except KeyboardInterrupt:

            print("\nGoodbye!")
            break

        except Exception as err:

            print(f"\nAI Error: {err}")


if __name__ == "__main__":
    main()