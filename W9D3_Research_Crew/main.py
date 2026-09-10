from crewai import Agent, Task, Crew, Process, LLM


# -----------------------------
# Local Ollama LLM
# -----------------------------
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434"
)


# -----------------------------
# Agent 1: Researcher
# -----------------------------
researcher = Agent(
    role="Researcher",
    goal="Research and collect accurate information about the given topic.",
    backstory=(
        "You are a careful AI researcher. "
        "You identify important facts, concepts, advantages, disadvantages, "
        "and practical examples related to a topic."
    ),
    llm=llm,
    verbose=True
)


# -----------------------------
# Agent 2: Writer
# -----------------------------
writer = Agent(
    role="Writer",
    goal="Convert research findings into a clear and well-structured article.",
    backstory=(
        "You are a professional technical writer. "
        "You explain technical topics in simple and understandable language "
        "while maintaining accuracy."
    ),
    llm=llm,
    verbose=True
)


# -----------------------------
# Agent 3: Reviewer
# -----------------------------
reviewer = Agent(
    role="Reviewer",
    goal="Review the written article for accuracy, clarity, completeness, and quality.",
    backstory=(
        "You are an experienced reviewer. "
        "You check content for incorrect information, missing points, "
        "poor structure, and unclear explanations."
    ),
    llm=llm,
    verbose=True
)


# -----------------------------
# Task 1: Research
# -----------------------------
research_task = Task(
    description=(
        "Research the topic: Artificial Intelligence in Healthcare. "
        "Provide important concepts, applications, benefits, challenges, "
        "and real-world examples. "
        "The research should be factual and organized."
    ),
    expected_output=(
        "A structured research summary containing key facts, "
        "applications, benefits, challenges, and examples."
    ),
    agent=researcher
)


# -----------------------------
# Task 2: Writing
# -----------------------------
writing_task = Task(
    description=(
        "Using the research provided by the Researcher, write a clear "
        "article about Artificial Intelligence in Healthcare. "
        "Include an introduction, major applications, benefits, challenges, "
        "and conclusion. Use simple professional language."
    ),
    expected_output=(
        "A well-structured article about Artificial Intelligence in Healthcare."
    ),
    agent=writer,
    context=[research_task]
)


# -----------------------------
# Task 3: Review
# -----------------------------
review_task = Task(
    description=(
        "Review the article produced by the Writer. "
        "Check it for factual accuracy, clarity, organization, completeness, "
        "and unnecessary repetition. "
        "Provide the final improved version of the article."
    ),
    expected_output=(
        "A reviewed and improved final article with accurate and clear content."
    ),
    agent=reviewer,
    context=[writing_task]
)


# -----------------------------
# Create Crew
# -----------------------------
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True
)


# -----------------------------
# Run Crew
# -----------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("W9D3 RESEARCH CREW")
    print("=" * 60)
    print("Researcher -> Writer -> Reviewer")
    print("LLM: Ollama llama3.2:3b")
    print("=" * 60 + "\n")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("FINAL REVIEWED OUTPUT")
    print("=" * 60)
    print(result)
    print("=" * 60)