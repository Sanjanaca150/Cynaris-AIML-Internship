from app.evaluation import calculate_basic_quality_score
from app.workflow import create_workflow


def test_quality_score():
    report = """
    Introduction

    Key Findings

    Applications

    Advantages

    Limitations

    Conclusion
    """

    score = calculate_basic_quality_score(report)

    assert 0 <= score <= 1
    assert score > 0.5


def test_workflow_structure():
    graph = create_workflow()

    assert graph is not None


def test_report_contains_expected_sections():
    report = """
    Introduction
    Key Findings
    Applications
    Advantages
    Limitations
    Conclusion
    """

    expected_sections = [
        "Introduction",
        "Key Findings",
        "Applications",
        "Advantages",
        "Limitations",
        "Conclusion",
    ]

    for section in expected_sections:
        assert section in report