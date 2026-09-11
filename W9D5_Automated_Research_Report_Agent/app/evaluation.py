"""
Evaluation utilities for the research report.

Ragas is used when available. A lightweight deterministic fallback
score is included so the project remains executable if Ragas evaluation
cannot be configured with the local model.
"""

import re


def calculate_basic_quality_score(report: str) -> float:
    """
    Calculate a simple deterministic quality score.
    """

    required_sections = [
        "Introduction",
        "Key Findings",
        "Applications",
        "Advantages",
        "Limitations",
        "Conclusion",
    ]

    section_score = sum(
        section.lower() in report.lower()
        for section in required_sections
    ) / len(required_sections)

    word_count = len(re.findall(r"\b\w+\b", report))

    length_score = min(word_count / 500, 1.0)

    score = (section_score * 0.7) + (length_score * 0.3)

    return round(score, 4)


def evaluate_with_ragas(topic: str, report: str) -> float:
    """
    Run a Ragas evaluation using the generated report.

    The evaluation uses a simple single-sample dataset containing
    the research topic as the user input and the generated report
    as the response.
    """

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import ResponseRelevancy

    sample = SingleTurnSample(
        user_input=topic,
        response=report,
    )

    dataset = EvaluationDataset(samples=[sample])

    result = evaluate(
        dataset=dataset,
        metrics=[ResponseRelevancy()],
    )

    scores = result.to_pandas()

    if "response_relevancy" not in scores.columns:
        raise RuntimeError(
            "Ragas did not return a response_relevancy score."
        )

    value = scores["response_relevancy"].iloc[0]

    return round(float(value), 4)


def evaluate_report(topic: str, report: str) -> dict:
    """
    Evaluate the generated report.

    Ragas is attempted first. If Ragas requires additional model
    configuration or fails during evaluation, the deterministic
    quality score is used as a fallback.
    """

    basic_score = calculate_basic_quality_score(report)

    try:
        import ragas

        ragas_available = True
        ragas_version = getattr(
            ragas,
            "__version__",
            "installed",
        )

        try:
            ragas_score = evaluate_with_ragas(
                topic=topic,
                report=report,
            )

            return {
                "topic": topic,
                "quality_score": ragas_score,
                "ragas_score": ragas_score,
                "basic_quality_score": basic_score,
                "ragas_available": True,
                "ragas_used": True,
                "ragas_version": ragas_version,
            }

        except Exception as exc:
            return {
                "topic": topic,
                "quality_score": basic_score,
                "ragas_score": None,
                "basic_quality_score": basic_score,
                "ragas_available": True,
                "ragas_used": False,
                "ragas_version": ragas_version,
                "ragas_error": str(exc),
            }

    except ImportError:
        return {
            "topic": topic,
            "quality_score": basic_score,
            "ragas_score": None,
            "basic_quality_score": basic_score,
            "ragas_available": False,
            "ragas_used": False,
            "ragas_version": "not installed",
        }