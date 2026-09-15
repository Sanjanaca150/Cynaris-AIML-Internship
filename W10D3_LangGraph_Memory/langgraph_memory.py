from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command


# =========================================================
# 1. DEFINE GRAPH STATE
# =========================================================

class AgentState(TypedDict, total=False):
    user_input: str
    classification: str
    route: str
    response: str
    human_feedback: str


# =========================================================
# 2. CLASSIFY NODE
# =========================================================

def classify_node(state: AgentState):
    user_input = state["user_input"].lower()

    if any(word in user_input for word in [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening"
    ]):
        classification = "greeting"

    elif any(word in user_input for word in [
        "error",
        "bug",
        "issue",
        "problem",
        "not working",
        "failed"
    ]):
        classification = "technical"

    else:
        classification = "general"

    print(f"[CLASSIFY] {classification}")

    return {
        "classification": classification
    }


# =========================================================
# 3. ROUTE NODE
# =========================================================

def route_node(state: AgentState):
    classification = state["classification"]

    if classification == "greeting":
        route = "friendly"

    elif classification == "technical":
        route = "technical_support"

    else:
        route = "general_response"

    print(f"[ROUTE] {route}")

    return {
        "route": route
    }


# =========================================================
# 4. CONDITIONAL ROUTING FUNCTION
# =========================================================

def conditional_route(state: AgentState):
    route = state["route"]

    if route == "friendly":
        return "friendly"

    elif route == "technical_support":
        return "technical"

    else:
        return "general"


# =========================================================
# 5. RESPOND NODE
# =========================================================

def respond_node(state: AgentState):
    route = state["route"]

    if route == "friendly":
        draft_response = (
            "Hello! Nice to meet you. "
            "How can I help you today?"
        )

    elif route == "technical_support":
        draft_response = (
            "I understand that you are facing a technical problem. "
            "Please provide the error message or describe the issue "
            "so that I can help troubleshoot it."
        )

    else:
        draft_response = (
            "Thanks for your message. "
            "I am ready to help with your question."
        )

    print("\n[HUMAN REVIEW REQUIRED]")
    print(f"Draft Response: {draft_response}")

    # -----------------------------------------------------
    # Human-in-the-loop interruption
    # -----------------------------------------------------

    human_feedback = interrupt({
        "message": "Human review required before final response.",
        "draft_response": draft_response,
        "instruction": (
            "Enter 'approve' to accept the draft, "
            "or enter a replacement response."
        )
    })

    # -----------------------------------------------------
    # Process human input
    # -----------------------------------------------------

    if isinstance(human_feedback, str):

        feedback = human_feedback.strip()

        if feedback.lower() in [
            "approve",
            "approved",
            "yes"
        ]:
            final_response = draft_response

        else:
            final_response = feedback

    else:
        final_response = draft_response

    print(f"[RESPOND] {final_response}")

    return {
        "response": final_response,
        "human_feedback": str(human_feedback)
    }


# =========================================================
# 6. BUILD LANGGRAPH
# =========================================================

builder = StateGraph(AgentState)

# Add the three required nodes
builder.add_node("classify", classify_node)
builder.add_node("route", route_node)
builder.add_node("respond", respond_node)

# START -> CLASSIFY
builder.add_edge(START, "classify")

# CLASSIFY -> ROUTE
builder.add_edge("classify", "route")

# ROUTE -> RESPOND using conditional routing
builder.add_conditional_edges(
    "route",
    conditional_route,
    {
        "friendly": "respond",
        "technical": "respond",
        "general": "respond"
    }
)

# RESPOND -> END
builder.add_edge("respond", END)


# =========================================================
# 7. PERSISTENT MEMORY CONFIGURATION
# =========================================================

DB_PATH = "langgraph_memory.db"


def create_graph():
    """
    Create the SQLite checkpointer and compile the graph.

    SqliteSaver.from_conn_string() returns a context manager,
    so we explicitly enter it before passing the saver to
    builder.compile().
    """

    checkpointer_context = SqliteSaver.from_conn_string(DB_PATH)

    checkpointer = checkpointer_context.__enter__()

    graph = builder.compile(
        checkpointer=checkpointer
    )

    return graph, checkpointer_context


# =========================================================
# 8. MAIN TEST PROGRAM
# =========================================================

if __name__ == "__main__":

    graph, checkpointer_context = create_graph()

    # The same thread ID is used to demonstrate
    # persistent conversation state.
    config = {
        "configurable": {
            "thread_id": "w10d3-demo"
        }
    }

    # Five required test inputs
    test_inputs = [
        "Hello, how are you?",
        "My application has an error.",
        "What is LangGraph?",
        "The API is not working.",
        "Thank you for your help."
    ]

    print("\n========================================")
    print("W10D3 LANGGRAPH + MEMORY DEMO")
    print("========================================")

    print("\nGraph:")
    print("START -> classify -> route -> respond -> END")

    print("\nThread ID:")
    print(config["configurable"]["thread_id"])

    print("\nPersistent database:")
    print(DB_PATH)

    # =====================================================
    # Run all five tests
    # =====================================================

    for number, user_input in enumerate(test_inputs, start=1):

        print("\n----------------------------------------")
        print(f"TEST {number}")
        print("----------------------------------------")

        print(f"User: {user_input}")

        # Start graph execution
        result = graph.invoke(
            {
                "user_input": user_input
            },
            config
        )

        # -------------------------------------------------
        # Graph pauses at interrupt()
        # -------------------------------------------------

        if "__interrupt__" in result:

            print("\n[GRAPH PAUSED]")
            print("Human-in-the-loop interruption detected.")

            human_input = input(
                "Human input (approve or enter replacement): "
            )

            # -------------------------------------------------
            # Resume graph with human input
            # -------------------------------------------------

            result = graph.invoke(
                Command(resume=human_input),
                config
            )

            print("\n[GRAPH RESUMED]")

        # -------------------------------------------------
        # Display final response
        # -------------------------------------------------

        print(f"Final Response: {result['response']}")

    # =====================================================
    # MEMORY VERIFICATION
    # =====================================================

    print("\n========================================")
    print("PERSISTENT MEMORY VERIFICATION")
    print("========================================")

    print(f"Thread ID: {config['configurable']['thread_id']}")
    print(f"SQLite database: {DB_PATH}")

    print("\nThe graph used the same thread ID for all five tests.")
    print("LangGraph checkpointing was enabled using SQLite.")

    # =====================================================
    # CLOSE SQLITE CHECKPOINTER
    # =====================================================

    checkpointer_context.__exit__(None, None, None)

    print("\n========================================")
    print("W10D3 COMPLETED SUCCESSFULLY")
    print("========================================")