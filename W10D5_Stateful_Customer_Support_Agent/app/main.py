from pathlib import Path

import mlflow

from app.crew_support import create_support_crew
from app.customer_support import build_support_graph


# Store MLflow data inside this project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# Use an explicit SQLite tracking URI.
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)


def run_customer_support():
    """Run the complete stateful customer support demonstration."""

    print("=" * 65)
    print("STATEFUL CUSTOMER SUPPORT AGENT")
    print("=" * 65)

    graph = build_support_graph()

    thread_id = "customer-1001"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    customer_name = "Sanjana"
    order_id = "ORD-1001"

    mlflow.set_experiment(
        "W10D5_Stateful_Customer_Support"
    )

    with mlflow.start_run():

        mlflow.log_param(
            "model",
            "llama3.2:3b"
        )

        mlflow.log_param(
            "framework",
            "LangGraph + CrewAI"
        )

        mlflow.log_param(
            "thread_id",
            thread_id
        )

        # --------------------------------------------------
        # TURN 1
        # --------------------------------------------------

        first_message = (
            "Hi, my name is Sanjana. "
            "My order ORD-1001 has not arrived yet."
        )

        first_state = {
            "messages": [first_message],
            "customer_name": customer_name,
            "order_id": order_id,
        }

        result_1 = graph.invoke(
            first_state,
            config
        )

        print("\nTURN 1")
        print("-" * 65)
        print("Customer:", first_message)
        print("Issue:", result_1.get("issue"))
        print("Agent:", result_1.get("response"))
        print("Status:", result_1.get("status"))

        mlflow.log_metric(
            "turn_1_response_length",
            len(result_1.get("response", ""))
        )

        # --------------------------------------------------
        # TURN 2
        # --------------------------------------------------

        second_message = (
            "I want a refund because I have been "
            "waiting too long."
        )

        second_state = {
            "messages": [second_message]
        }

        result_2 = graph.invoke(
            second_state,
            config
        )

        print("\nTURN 2")
        print("-" * 65)
        print("Customer:", second_message)
        print(
            "Remembered customer:",
            result_2.get("customer_name")
        )
        print(
            "Remembered order:",
            result_2.get("order_id")
        )
        print("Issue:", result_2.get("issue"))
        print("Agent:", result_2.get("response"))
        print("Status:", result_2.get("status"))

        mlflow.log_metric(
            "turn_2_response_length",
            len(result_2.get("response", ""))
        )

        mlflow.log_metric(
            "human_escalation",
            int(result_2.get("needs_human", False))
        )

        # --------------------------------------------------
        # CREWAI
        # --------------------------------------------------

        print("\nCREWAI SUPPORT ANALYSIS")
        print("-" * 65)

        crew = create_support_crew(
            result_2.get(
                "customer_name",
                customer_name
            ),
            result_2.get(
                "issue",
                ""
            ),
        )

        try:
            crew_result = crew.kickoff()
            print(crew_result)

            mlflow.log_param(
                "crewai_status",
                "success"
            )

        except Exception as exc:
            print(
                "CrewAI analysis failed:",
                exc
            )

            mlflow.log_param(
                "crewai_status",
                "failed"
            )

        # --------------------------------------------------
        # MLFLOW
        # --------------------------------------------------

        mlflow.set_tag(
            "workflow_status",
            result_2.get(
                "status",
                "Unknown"
            )
        )

        mlflow.set_tag(
            "mlops_tracking",
            "enabled"
        )

        mlflow.set_tag(
            "stateful_memory",
            "MemorySaver"
        )

        print("\n" + "=" * 65)
        print("MLFLOW TRACKING COMPLETED")
        print("=" * 65)
        print(
            "Tracking database:",
            MLFLOW_DB
        )
        print("=" * 65)


if __name__ == "__main__":
    run_customer_support()