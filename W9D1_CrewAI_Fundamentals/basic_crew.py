from crewai import Agent, Crew, LLM, Process, Task


# --------------------------------------------------
# Local Ollama LLM
# --------------------------------------------------

llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
)


# --------------------------------------------------
# Agents
# --------------------------------------------------

researcher = Agent(
    role="Researcher",
    goal=(
        "Research the assigned topic and identify accurate, "
        "relevant, and useful information."
    ),
    backstory=(
        "You are a careful AI research assistant. "
        "You collect important facts, organize information "
        "clearly, and avoid unsupported claims."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


writer = Agent(
    role="Writer",
    goal=(
        "Transform research findings into a clear, "
        "well-structured and easy-to-understand explanation."
    ),
    backstory=(
        "You are a technical writer who explains complex "
        "AI concepts using simple and organized language."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


reviewer = Agent(
    role="Reviewer",
    goal=(
        "Review the written content for accuracy, clarity, "
        "completeness, relevance, and organization."
    ),
    backstory=(
        "You are a quality reviewer who checks technical "
        "content carefully and improves weaknesses before "
        "the final answer is delivered."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# --------------------------------------------------
# Tasks
# --------------------------------------------------

research_task = Task(
    description=(
        "Research the topic 'How multi-agent AI systems work'. "
        "Identify the main concepts, explain how multiple agents "
        "can collaborate, and describe their roles, advantages, "
        "challenges, and practical applications. "
        "Provide concise and organized research notes for the Writer."
    ),
    expected_output=(
        "Structured research notes covering the definition of "
        "multi-agent AI systems, agent collaboration, common roles, "
        "advantages, challenges, and practical applications."
    ),
    agent=researcher,
)


writing_task = Task(
    description=(
        "Use the Researcher's findings to write a clear explanation "
        "of how multi-agent AI systems work. Include headings for "
        "the concept, agent collaboration, benefits, challenges, "
        "and applications. Use simple technical language."
    ),
    expected_output=(
        "A well-structured explanation of multi-agent AI systems "
        "with clear headings, accurate information, and practical examples."
    ),
    agent=writer,
)


review_task = Task(
    description=(
        "Review the Writer's explanation carefully. Check whether "
        "the information is accurate, relevant, complete, clear, "
        "and logically organized. Correct weaknesses and produce "
        "a polished final version."
    ),
    expected_output=(
        "A reviewed and improved final explanation that is accurate, "
        "complete, clear, relevant, and well organized."
    ),
    agent=reviewer,
)


# --------------------------------------------------
# Crew
# --------------------------------------------------

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


# --------------------------------------------------
# Run the Crew
# --------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("W9D1 - CREWAI FUNDAMENTALS")
    print("Multi-Agent Research Crew")
    print("=" * 70)

    print("\nStarting Researcher -> Writer -> Reviewer workflow...\n")

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL CREW OUTPUT")
    print("=" * 70)
    print(result)

    print("\n" + "=" * 70)
    print("W9D1 BASIC CREW EXECUTION COMPLETED")
    print("=" * 70)