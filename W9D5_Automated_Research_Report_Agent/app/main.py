"""
Entry point for the Automated Research Report Agent.
"""

from datetime import datetime
from pathlib import Path

import mlflow


from .evaluation import evaluate_report
from .workflow import create_workflow


# Directory where generated research reports are saved
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


# Local SQLite database used by MLflow
MLFLOW_DB = Path("mlflow.db").resolve()


def run_research(topic: str):
    """
    Run the automated research workflow and track the execution
    using MLflow.
    """

    # Use SQLite as the MLflow tracking backend.
    # This avoids the deprecated filesystem backend problem.
    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    # Create or use the MLflow experiment.
    mlflow.set_experiment(
        "W9D5_Automated_Research_Report_Agent"
    )

    # Start an MLflow run
    with mlflow.start_run(
        run_name="automated-research-report"
    ) as run:

        # Log project information
        mlflow.log_param("topic", topic)
        mlflow.log_param("crew_agents", 3)
        mlflow.log_param(
            "architecture",
            "LangGraph -> CrewAI -> Ragas -> MLflow",
        )

        # Create the LangGraph workflow
        graph = create_workflow()

        # Execute the workflow
        result = graph.invoke(
            {
                "topic": topic,
                "status": "started",
            }
        )

        # Get the generated report
        report = result["report"]

        # Evaluate the report
        evaluation = evaluate_report(
            topic=topic,
            report=report,
        )

        # Log evaluation metric
        mlflow.log_metric(
            "quality_score",
            evaluation["quality_score"],
        )

        # Log Ragas information
        mlflow.log_param(
            "ragas_available",
            str(evaluation["ragas_available"]),
        )

        mlflow.log_param(
            "ragas_version",
            evaluation["ragas_version"],
        )

        # Create unique report filename
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        report_path = (
            REPORT_DIR
            / f"research_report_{timestamp}.txt"
        )

        # Save generated report
        report_path.write_text(
            report,
            encoding="utf-8",
        )

        # Save report as MLflow artifact
        mlflow.log_artifact(
            str(report_path)
        )

        # Display the final report
        print("\n" + "=" * 70)
        print("AUTOMATED RESEARCH REPORT")
        print("=" * 70)

        print(report)

        # Display evaluation details
        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)

        print(
            f"Quality Score: "
            f"{evaluation['quality_score']}"
        )

        print(
            f"Ragas Available: "
            f"{evaluation['ragas_available']}"
        )

        print(
            f"Ragas Version: "
            f"{evaluation['ragas_version']}"
        )

        print(
            f"MLflow Run ID: "
            f"{run.info.run_id}"
        )

        print(
            f"Report Saved: "
            f"{report_path}"
        )

        print(
            f"MLflow Database: "
            f"{MLFLOW_DB}"
        )

        print("=" * 70)

        return result


if __name__ == "__main__":

    # Research topic for the W9D5 project
    research_topic = (
        "Applications and challenges of Generative AI "
        "in software engineering"
    )

    # Start the automated research process
    run_research(research_topic)