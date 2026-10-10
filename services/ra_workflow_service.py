import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from ra_agent import ResearchAssistant


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def should_continue(state: MessageState):
    """Choose whether to execute tools or finish the workflow."""

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", []):
        return "use_tool"

    return END


class RAWorkflowService:

    def __init__(self, mcp_tools):

        self.ra_agent = ResearchAssistant(
            mcp_tools=mcp_tools
        )

        workflow = StateGraph(MessageState)

        workflow.add_node(
            "research_node",
            self.ra_agent.call_llm
        )

        workflow.add_node(
            "research_tool_node",
            self.ra_agent.tool_node
        )

        workflow.add_edge(
            START,
            "research_node"
        )

        workflow.add_conditional_edges(
            "research_node",
            should_continue,
            {
                "use_tool": "research_tool_node",
                END: END,
            }
        )

        workflow.add_edge(
            "research_tool_node",
            "research_node"
        )

        self.workflow = workflow.compile()

    async def run_ra_workflow(self, query: str):
        """Run the research workflow for a user query."""

        result = await self.workflow.ainvoke(
            {
                "messages": [
                    HumanMessage(content=query)
                ],
                "llm_calls": 0
            },
            config={
                "recursion_limit": 18
            }
        )

        return result