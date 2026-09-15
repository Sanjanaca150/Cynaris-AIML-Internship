from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------

class AgentState(TypedDict):
    user_input: str
    category: str
    route: str
    response: str


# ---------------------------------------------------------
# NODE 1: CLASSIFY
# ---------------------------------------------------------

def classify(state: AgentState):
    user_input = state["user_input"].lower()

    if any(word in user_input for word in ["python", "java", "sql", "coding", "programming"]):
        category = "technical"

    elif any(word in user_input for word in ["hello", "hi", "hey", "good morning"]):
        category = "general"

    elif any(word in user_input for word in ["job", "career", "interview", "resume"]):
        category = "career"

    else:
        category = "general"

    print(f"[CLASSIFY] Input: {state['user_input']}")
    print(f"[CLASSIFY] Category: {category}")

    return {"category": category}


# ---------------------------------------------------------
# NODE 2: ROUTE
# ---------------------------------------------------------

def route(state: AgentState):
    category = state["category"]

    if category == "technical":
        route_name = "technical"
    elif category == "career":
        route_name = "career"
    else:
        route_name = "general"

    print(f"[ROUTE] Routing to: {route_name}")

    return {"route": route_name}


# ---------------------------------------------------------
# NODE 3: RESPOND
# ---------------------------------------------------------

def respond(state: AgentState):
    route_name = state["route"]

    if route_name == "technical":
        response = (
            "Technical response: I can help with Python, Java, SQL, "
            "programming and AI/ML topics."
        )

    elif route_name == "career":
        response = (
            "Career response: I can help with interviews, resumes, "
            "career planning and job preparation."
        )

    else:
        response = (
            "General response: Hello! I can help you with general questions."
        )

    print(f"[RESPOND] {response}")

    return {"response": response}


# ---------------------------------------------------------
# CONDITIONAL ROUTING
# ---------------------------------------------------------

def conditional_route(state: AgentState):
    return state["route"]


# ---------------------------------------------------------
# BUILD THREE-NODE GRAPH
# ---------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)

builder.add_edge(START, "classify")
builder.add_edge("classify", "route")

builder.add_conditional_edges(
    "route",
    conditional_route,
    {
        "technical": "respond",
        "career": "respond",
        "general": "respond",
    },
)

builder.add_edge("respond", END)

graph = builder.compile()


# ---------------------------------------------------------
# TEST 5 INPUTS
# ---------------------------------------------------------

def test_five_inputs():
    print("\n" + "=" * 60)
    print("TESTING THREE-NODE LANGGRAPH")
    print("=" * 60)

    test_inputs = [
        "I want to learn Python",
        "I have a Java programming question",
        "How should I prepare for an interview?",
        "Hello, good morning",
        "I need help with SQL",
    ]

    for number, user_input in enumerate(test_inputs, start=1):
        print(f"\n--- TEST {number} ---")

        result = graph.invoke({
            "user_input": user_input
        })

        print(f"Final Category : {result['category']}")
        print(f"Final Route    : {result['route']}")
        print(f"Final Response : {result['response']}")


# ---------------------------------------------------------
# HUMAN-IN-THE-LOOP GRAPH
# ---------------------------------------------------------

def human_review(state: AgentState):
    print("\n[HUMAN REVIEW] Graph paused.")
    print("[HUMAN REVIEW] Waiting for human input...")

    human_decision = interrupt({
        "message": "Please review the response and enter APPROVE or EDIT.",
        "current_response": state["response"],
    })

    print(f"[HUMAN REVIEW] Human input: {human_decision}")

    if isinstance(human_decision, str):
        if human_decision.upper() == "APPROVE":
            return {
                "response": state["response"] + " [Approved by human]"
            }
        else:
            return {
                "response": human_decision
            }

    return state


# Build HITL graph
hitl_builder = StateGraph(AgentState)

hitl_builder.add_node("classify", classify)
hitl_builder.add_node("route", route)
hitl_builder.add_node("respond", respond)
hitl_builder.add_node("human_review", human_review)

hitl_builder.add_edge(START, "classify")
hitl_builder.add_edge("classify", "route")

hitl_builder.add_conditional_edges(
    "route",
    conditional_route,
    {
        "technical": "respond",
        "career": "respond",
        "general": "respond",
    },
)

hitl_builder.add_edge("respond", "human_review")
hitl_builder.add_edge("human_review", END)

memory = MemorySaver()

hitl_graph = hitl_builder.compile(checkpointer=memory)


# ---------------------------------------------------------
# TEST HUMAN-IN-THE-LOOP
# ---------------------------------------------------------

def test_human_in_loop():
    print("\n" + "=" * 60)
    print("TESTING HUMAN-IN-THE-LOOP INTERRUPT")
    print("=" * 60)

    config = {
        "configurable": {
            "thread_id": "w10d1-demo"
        }
    }

    initial_state = {
        "user_input": "I want to learn Python"
    }

    result = hitl_graph.invoke(
        initial_state,
        config=config
    )

    print("\nGraph execution paused for human review.")
    print("Current state:")
    print(result)

    print("\nResuming graph with human decision: APPROVE")

    resumed = hitl_graph.invoke(
        Command(resume="APPROVE"),
        config=config
    )

    print("\nGraph resumed successfully.")
    print("Final result:")
    print(resumed)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    test_five_inputs()
    test_human_in_loop()