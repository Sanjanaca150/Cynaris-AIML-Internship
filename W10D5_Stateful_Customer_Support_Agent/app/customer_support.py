from typing import List, TypedDict

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class SupportState(TypedDict, total=False):
    """State maintained throughout a customer support conversation."""

    messages: List[str]
    customer_name: str
    order_id: str
    issue: str
    response: str
    status: str
    needs_human: bool


# Local Ollama LLM.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


def analyze_request(state: SupportState):
    """Analyze the customer's latest request."""

    messages = state.get("messages", [])
    latest_message = messages[-1] if messages else ""

    prompt = f"""
You are a customer support assistant.

Customer name: {state.get("customer_name", "Unknown")}
Order ID: {state.get("order_id", "Unknown")}

Analyze ONLY the latest customer message.

Latest message:
{latest_message}

Identify the customer's current request or issue in one short sentence.
Do not repeat an older issue if the customer has made a new request.
"""

    result = llm.invoke(prompt)

    return {
        "issue": result.content.strip()
    }


def check_escalation(state: SupportState):
    """Check whether the latest request requires human assistance."""

    messages = state.get("messages", [])
    latest_message = messages[-1].lower() if messages else ""

    escalation_keywords = [
        "refund",
        "cancel",
        "fraud",
        "unauthorized",
        "complaint",
        "legal",
        "damaged",
        "payment dispute",
    ]

    needs_human = any(
        keyword in latest_message
        for keyword in escalation_keywords
    )

    return {
        "needs_human": needs_human
    }


def generate_response(state: SupportState):
    """Generate the final customer-facing response."""

    customer_name = state.get("customer_name", "Customer")
    order_id = state.get("order_id", "Not provided")
    issue = state.get("issue", "Not provided")

    if state.get("needs_human"):
        response = (
            f"Thank you, {customer_name}. "
            f"Your request regarding order {order_id} requires "
            "additional assistance. I have marked it for human "
            "support review."
        )

        status = "Escalated"

    else:
        prompt = f"""
You are a professional customer support assistant.

Customer: {customer_name}
Order ID: {order_id}
Current issue: {issue}

Provide a concise and helpful response.
Do not invent order details, tracking information, refunds,
or company policies.
"""

        result = llm.invoke(prompt)

        response = result.content.strip()
        status = "Resolved"

    return {
        "response": response,
        "status": status,
    }


def build_support_graph():
    """Build and compile the stateful LangGraph workflow."""

    graph = StateGraph(SupportState)

    graph.add_node("analyze_request", analyze_request)
    graph.add_node("check_escalation", check_escalation)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "analyze_request")
    graph.add_edge("analyze_request", "check_escalation")
    graph.add_edge("check_escalation", "generate_response")
    graph.add_edge("generate_response", END)

    # MemorySaver persists state for each thread_id.
    memory = MemorySaver()

    return graph.compile(checkpointer=memory)