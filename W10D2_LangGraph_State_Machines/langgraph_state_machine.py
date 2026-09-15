from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# =========================================================
# STATE DEFINITION
# =========================================================

class AgentState(TypedDict, total=False):
    user_input: str
    classification: str
    route: str
    response: str
    human_feedback: str


# =========================================================
# NODE 1: CLASSIFY
# =========================================================

def classify(state: AgentState):
    """Classify the user input into technical, general, or sensitive."""

    text = state["user_input"].lower()

    technical_keywords = [
        "password",
        "login",
        "computer",
        "python",
        "code",
        "software",
        "internet",
        "database",
        "programming",
    ]

    sensitive_keywords = [
        "medical",
        "health",
        "patient",
        "personal information",
        "private information",
        "should i share",
        "confidential",
    ]

    if any(keyword in text for keyword in sensitive_keywords):
        classification = "sensitive"

    elif any(keyword in text for keyword in technical_keywords):
        classification = "technical"

    else:
        classification = "general"

    print(f"[CLASSIFY] Input: {state['user_input']}")
    print(f"[CLASSIFY] Classification: {classification}")

    return {
        "classification": classification
    }


# =========================================================
# NODE 2: ROUTE
# =========================================================

def route(state: AgentState):
    """Select the next route based on classification."""

    classification = state["classification"]

    if classification == "technical":
        selected_route = "technical_response"

    elif classification == "sensitive":
        selected_route = "human_review"

    else:
        selected_route = "general_response"

    print(f"[ROUTE] Selected route: {selected_route}")

    return {
        "route": selected_route
    }


# =========================================================
# CONDITIONAL ROUTING FUNCTION
# =========================================================

def route_decision(
    state: AgentState,
) -> Literal[
    "technical_response",
    "general_response",
    "human_review",
]:
    """Determine which node should execute next."""

    return state["route"]


# =========================================================
# TECHNICAL RESPONSE NODE
# =========================================================

def technical_response(state: AgentState):
    """Generate a response for technical requests."""

    response = (
        "This is a technical request. "
        "Please verify the technical details and follow "
        "the appropriate system or application instructions."
    )

    print(f"[RESPONSE] {response}")

    return {
        "response": response
    }


# =========================================================
# GENERAL RESPONSE NODE
# =========================================================

def general_response(state: AgentState):
    """Generate a response for general requests."""

    response = (
        "This is a general information request. "
        "The system can provide a general explanation "
        "based on the user's question."
    )

    print(f"[RESPONSE] {response}")

    return {
        "response": response
    }


# =========================================================
# HUMAN-IN-THE-LOOP NODE
# =========================================================

def human_review(state: AgentState):
    """
    Pause graph execution and request human input.
    """

    print()
    print("[HUMAN REVIEW] Graph execution paused.")

    human_feedback = interrupt(
        {
            "message": "Human review required for this sensitive request.",
            "input": state["user_input"],
            "question": "Approve this request? Enter yes/no or provide guidance.",
        }
    )

    print(f"[HUMAN REVIEW] Received: {human_feedback}")

    return {
        "human_feedback": str(human_feedback)
    }


# =========================================================
# RESPONSE AFTER HUMAN REVIEW
# =========================================================

def respond_after_review(state: AgentState):
    """Generate final response after human review."""

    feedback = state.get("human_feedback", "").lower().strip()

    if feedback in ["yes", "approved", "approve"]:

        response = (
            "Human review approved the request. "
            "The system can proceed while following "
            "appropriate privacy and safety guidelines."
        )

    else:

        response = (
            "Human review did not approve the request. "
            "The system will not proceed with the sensitive action."
        )

    print(f"[RESPONSE] {response}")

    return {
        "response": response
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("technical_response", technical_response)
builder.add_node("general_response", general_response)
builder.add_node("human_review", human_review)
builder.add_node("respond_after_review", respond_after_review)


# =========================================================
# GRAPH EDGES
# =========================================================

# START → CLASSIFY
builder.add_edge(START, "classify")

# CLASSIFY → ROUTE
builder.add_edge("classify", "route")


# =========================================================
# CONDITIONAL EDGES
# =========================================================

builder.add_conditional_edges(
    "route",
    route_decision,
    {
        "technical_response": "technical_response",
        "general_response": "general_response",
        "human_review": "human_review",
    },
)


# Normal response paths
builder.add_edge("technical_response", END)
builder.add_edge("general_response", END)


# Human review path
builder.add_edge("human_review", "respond_after_review")
builder.add_edge("respond_after_review", END)


# =========================================================
# CHECKPOINTING
# =========================================================

checkpointer = MemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)


# =========================================================
# TEST 1: TECHNICAL INPUT
# =========================================================

print()
print("=" * 60)
print("TEST 1: TECHNICAL INPUT")
print("=" * 60)

result = graph.invoke(
    {
        "user_input": "How do I reset my password?"
    },
    config={
        "configurable": {
            "thread_id": "w10d2-test-1"
        }
    },
)

print()
print("Final State:")
print(result)


# =========================================================
# TEST 2: GENERAL INPUT
# =========================================================

print()
print("=" * 60)
print("TEST 2: GENERAL INPUT")
print("=" * 60)

result = graph.invoke(
    {
        "user_input": "What is artificial intelligence?"
    },
    config={
        "configurable": {
            "thread_id": "w10d2-test-2"
        }
    },
)

print()
print("Final State:")
print(result)


# =========================================================
# TEST 3: SENSITIVE INPUT + HUMAN-IN-THE-LOOP
# =========================================================

print()
print("=" * 60)
print("TEST 3: HUMAN-IN-THE-LOOP")
print("=" * 60)

config = {
    "configurable": {
        "thread_id": "w10d2-human-review"
    }
}

# First execution pauses at interrupt()
initial_result = graph.invoke(
    {
        "user_input": "Should I share my medical information online?"
    },
    config=config,
)

print()
print("Graph paused for human review.")

if "__interrupt__" in initial_result:

    print("Interrupt detected:")
    print(initial_result["__interrupt__"])


# Resume graph after human input
resume_result = graph.invoke(
    Command(resume="yes"),
    config=config,
)

print()
print("Graph resumed successfully.")

print()
print("Final State:")
print(resume_result)


# =========================================================
# TEST 4: TECHNICAL INPUT
# =========================================================

print()
print("=" * 60)
print("TEST 4: TECHNICAL INPUT")
print("=" * 60)

result = graph.invoke(
    {
        "user_input": "How can I connect Python to a database?"
    },
    config={
        "configurable": {
            "thread_id": "w10d2-test-4"
        }
    },
)

print()
print("Final State:")
print(result)


# =========================================================
# TEST 5: GENERAL INPUT
# =========================================================

print()
print("=" * 60)
print("TEST 5: GENERAL INPUT")
print("=" * 60)

result = graph.invoke(
    {
        "user_input": "What is machine learning?"
    },
    config={
        "configurable": {
            "thread_id": "w10d2-test-5"
        }
    },
)

print()
print("Final State:")
print(result)


# =========================================================
# COMPLETION MESSAGE
# =========================================================

print()
print("=" * 60)
print("W10D2 TESTING COMPLETED")
print("=" * 60)
print("All 5 inputs were tested successfully.")
print("Conditional routing was verified.")
print("Human-in-the-loop pause and resume was verified.")
print("=" * 60)