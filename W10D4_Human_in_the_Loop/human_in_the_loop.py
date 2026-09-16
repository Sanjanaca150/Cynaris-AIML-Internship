from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ---------------------------------------------------------
# State definition
# ---------------------------------------------------------
class AgentState(TypedDict, total=False):
    user_input: str
    classification: str
    route: str
    response: str
    human_decision: str


# ---------------------------------------------------------
# Node 1: Classify the input
# ---------------------------------------------------------
def classify(state: AgentState):
    user_input = state["user_input"].lower()

    if any(word in user_input for word in ["code", "python", "java", "program", "bug"]):
        classification = "technical"
    elif any(word in user_input for word in ["hello", "hi", "hey", "good morning"]):
        classification = "greeting"
    elif any(word in user_input for word in ["help", "problem", "issue", "error"]):
        classification = "support"
    else:
        classification = "general"

    print(f"[CLASSIFY] Input: {state['user_input']}")
    print(f"[CLASSIFY] Classification: {classification}")

    return {"classification": classification}


# ---------------------------------------------------------
# Node 2: Route the classified input
# ---------------------------------------------------------
def route(state: AgentState):
    classification = state["classification"]

    route_map = {
        "technical": "technical",
        "greeting": "greeting",
        "support": "support",
        "general": "general",
    }

    selected_route = route_map.get(classification, "general")

    print(f"[ROUTE] Classification: {classification}")
    print(f"[ROUTE] Selected route: {selected_route}")

    return {"route": selected_route}


# ---------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------
def route_decision(state: AgentState):
    return state["route"]


# ---------------------------------------------------------
# Node 3: Human-in-the-loop + response
# ---------------------------------------------------------
def respond(state: AgentState):
    route = state["route"]

    print(f"[RESPOND] Route received: {route}")

    # Pause execution and request human approval/input.
    human_input = interrupt(
        {
            "message": "Human approval required before generating the final response.",
            "route": route,
            "question": f"Do you approve the '{route}' route? Enter APPROVE or CHANGE.",
        }
    )

    human_decision = str(human_input).strip().upper()

    if human_decision == "APPROVE":
        response = (
            f"Human approved the {route} route. "
            f"Final response generated successfully."
        )
    else:
        response = (
            f"Human requested a change for the {route} route. "
            f"Response generation was reviewed by a human."
        )

    print(f"[HUMAN] Decision: {human_decision}")
    print(f"[RESPOND] {response}")

    return {
        "human_decision": human_decision,
        "response": response,
    }


# ---------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)

builder.add_edge(START, "classify")
builder.add_edge("classify", "route")

builder.add_conditional_edges(
    "route",
    route_decision,
    {
        "technical": "respond",
        "greeting": "respond",
        "support": "respond",
        "general": "respond",
    },
)

builder.add_edge("respond", END)


# ---------------------------------------------------------
# Memory/checkpoint required for interrupt + resume
# ---------------------------------------------------------
memory = MemorySaver()

graph = builder.compile(checkpointer=memory)


# ---------------------------------------------------------
# Test 1: Human-in-the-loop pause and resume
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("HUMAN-IN-THE-LOOP TEST")
print("=" * 60)

test_input = "I have a Python bug in my program."

config = {
    "configurable": {
        "thread_id": "w10d4-human-test"
    }
}

print(f"\nInput: {test_input}")
print("Starting graph...")

result = graph.invoke(
    {"user_input": test_input},
    config=config,
)

print("\nGraph paused for human input.")
print("Interrupt information:")

state = graph.get_state(config)

if state.interrupts:
    print(state.interrupts[0].value)

human_decision = input("\nEnter human decision (APPROVE/CHANGE): ")

print("\nResuming graph...")

final_result = graph.invoke(
    Command(resume=human_decision),
    config=config,
)

print("\nFinal Result:")
print(final_result)


# ---------------------------------------------------------
# Test 2–6: Five different inputs for routing verification
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("FIVE INPUT ROUTING TEST")
print("=" * 60)

test_inputs = [
    "Write a Python program to calculate factorial.",
    "Hello, good morning!",
    "I have a problem with my account.",
    "What is artificial intelligence?",
    "I found an error in my Java code.",
]

for number, text in enumerate(test_inputs, start=1):
    print(f"\n--- Test {number} ---")
    print(f"Input: {text}")

    # Run classification and routing only.
    classification_result = graph.invoke(
        {"user_input": text},
        config={
            "configurable": {
                "thread_id": f"routing-test-{number}"
            }
        },
        interrupt_before=["respond"],
    )

    print(f"Classification: {classification_result.get('classification')}")
    print(f"Route: {classification_result.get('route')}")
    print("Routing verified successfully.")