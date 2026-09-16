from app.customer_support import build_support_graph


def test_stateful_conversation():
    """Verify that customer information persists between turns."""

    graph = build_support_graph()

    config = {
        "configurable": {
            "thread_id": "test-customer-001"
        }
    }

    first_result = graph.invoke(
        {
            "messages": [
                "My name is Alex and my order is ORD-2001."
            ],
            "customer_name": "Alex",
            "order_id": "ORD-2001",
        },
        config,
    )

    assert first_result["customer_name"] == "Alex"
    assert first_result["order_id"] == "ORD-2001"

    second_result = graph.invoke(
        {
            "messages": [
                "I need help with my order."
            ]
        },
        config,
    )

    assert second_result["customer_name"] == "Alex"
    assert second_result["order_id"] == "ORD-2001"


def test_refund_escalation():
    """Verify that refund requests are escalated to human support."""

    graph = build_support_graph()

    config = {
        "configurable": {
            "thread_id": "test-customer-002"
        }
    }

    result = graph.invoke(
        {
            "messages": [
                "I want a refund for my order."
            ],
            "customer_name": "Test User",
            "order_id": "ORD-3001",
        },
        config,
    )

    assert result["needs_human"] is True
    assert result["status"] == "Escalated"