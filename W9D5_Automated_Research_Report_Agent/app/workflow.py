"""
LangGraph workflow for the Automated Research Report Agent.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from .agents import create_research_crew


class ResearchState(TypedDict, total=False):
    topic: str
    report: str
    status: str


def research_node(state: ResearchState):
    """Execute the CrewAI research crew."""

    topic = state["topic"]

    crew = create_research_crew(topic)

    result = crew.kickoff()

    return {
        "report": str(result),
        "status": "completed",
    }


def create_workflow():
    """Create and compile the LangGraph workflow."""

    builder = StateGraph(ResearchState)

    builder.add_node("research_crew", research_node)

    builder.add_edge(START, "research_crew")
    builder.add_edge("research_crew", END)

    return builder.compile()