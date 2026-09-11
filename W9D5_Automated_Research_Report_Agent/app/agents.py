"""
CrewAI agents for the Automated Research Report Agent.
"""

from crewai import Agent, Crew, Process, Task

from .tools import WebSearchTool


# Ollama model configuration
OLLAMA_MODEL = "ollama/llama3.2:3b"


def create_research_crew(topic: str):
    """
    Build a three-agent automated research crew.

    Agents:
    1. Researcher - searches for information
    2. Writer - creates the report
    3. Reviewer - reviews the report
    """

    # Create the web search tool
    search_tool = WebSearchTool()

    # ---------------------------------------------------------
    # Agent 1: Researcher
    # ---------------------------------------------------------
    researcher = Agent(
        role="Researcher",
        goal=(
            f"Research the topic '{topic}' and collect "
            "relevant, accurate and useful information."
        ),
        backstory=(
            "You are a careful AI research analyst. "
            "You search for relevant information, identify "
            "important facts and organize research findings "
            "clearly."
        ),
        tools=[search_tool],
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    # ---------------------------------------------------------
    # Agent 2: Writer
    # ---------------------------------------------------------
    writer = Agent(
        role="Technical Writer",
        goal=(
            "Convert the research findings into a clear, "
            "structured and informative technical report."
        ),
        backstory=(
            "You are an experienced technical writer. "
            "You transform research information into "
            "well-organized reports that are easy to understand."
        ),
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    # ---------------------------------------------------------
    # Agent 3: Reviewer
    # ---------------------------------------------------------
    reviewer = Agent(
        role="Reviewer",
        goal=(
            "Review the generated report for clarity, "
            "completeness, consistency and unsupported claims."
        ),
        backstory=(
            "You are a strict quality reviewer. "
            "You identify weak sections, missing information "
            "and unclear statements and improve the final report."
        ),
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    # ---------------------------------------------------------
    # Task 1: Research
    # ---------------------------------------------------------
    research_task = Task(
        description=(
            f"Research the following topic:\n\n"
            f"{topic}\n\n"
            "Use the web search tool to collect relevant "
            "information. Identify important facts, recent "
            "developments, practical applications, advantages "
            "and limitations. Organize the findings into "
            "structured research notes."
        ),
        expected_output=(
            "Detailed and structured research notes containing "
            "important findings, facts and source information."
        ),
        agent=researcher,
    )

    # ---------------------------------------------------------
    # Task 2: Writing
    # ---------------------------------------------------------
    writing_task = Task(
        description=(
            "Using the research notes produced by the Researcher, "
            "write a structured technical research report.\n\n"
            "The report must contain these sections:\n"
            "1. Introduction\n"
            "2. Key Findings\n"
            "3. Applications\n"
            "4. Advantages\n"
            "5. Limitations\n"
            "6. Conclusion\n\n"
            "Use clear and factual language."
        ),
        expected_output=(
            "A complete and well-structured technical research report."
        ),
        agent=writer,
        context=[research_task],
    )

    # ---------------------------------------------------------
    # Task 3: Review
    # ---------------------------------------------------------
    review_task = Task(
        description=(
            "Review the research report produced by the Writer.\n\n"
            "Check the report for:\n"
            "- factual consistency\n"
            "- clarity\n"
            "- completeness\n"
            "- logical structure\n"
            "- unsupported claims\n\n"
            "Correct weak sections and produce the final improved "
            "research report."
        ),
        expected_output=(
            "A reviewed, corrected and improved final research report."
        ),
        agent=reviewer,
        context=[writing_task],
    )

    # ---------------------------------------------------------
    # Create the Crew
    # ---------------------------------------------------------
    crew = Crew(
        agents=[
            researcher,
            writer,
            reviewer,
        ],
        tasks=[
            research_task,
            writing_task,
            review_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew