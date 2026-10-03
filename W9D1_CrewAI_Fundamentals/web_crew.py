from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Local Ollama LLM
# --------------------------------------------------

llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
)


# --------------------------------------------------
# Web Search Tool
# --------------------------------------------------

search_tool = SerperDevTool()


# --------------------------------------------------
# Agents
# --------------------------------------------------

researcher = Agent(
    role="Web Researcher",
    goal=(
        "Research the assigned topic using current web information "
        "and provide accurate, relevant, source-based findings."
    ),
    backstory=(
        "You are a careful web research specialist. "
        "You search the internet for reliable information, "
        "identify important facts, and organize findings clearly. "
        "You avoid unsupported claims and include useful sources."
    ),
    llm=llm,
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
)


writer = Agent(
    role="Technical Writer",
    goal=(
        "Transform the research findings into a clear, structured "
        "and easy-to-understand technical explanation."
    ),
    backstory=(
        "You are an experienced technical writer who converts "
        "research findings into accurate and readable content. "
        "You preserve important facts and source information."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


reviewer = Agent(
    role="Content Reviewer",
    goal=(
        "Review the final content for accuracy, relevance, clarity, "
        "completeness, and proper use of web-based information."
    ),
    backstory=(
        "You are a quality reviewer who carefully checks technical "
        "content and improves weak or unsupported statements before "
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
        "Research the topic 'Current applications of multi-agent AI systems'. "
        "Use the web search tool to find current information from reliable "
        "web sources. Identify at least three important applications, "
        "explain how multi-agent systems are used in those applications, "
        "and provide relevant source information for the findings."
    ),
    expected_output=(
        "Structured web-based research notes containing current information "
        "about at least three applications of multi-agent AI systems, "
        "with explanations and source information."
    ),
    agent=researcher,
)


writing_task = Task(
    description=(
        "Use the Researcher's web-based findings to write a clear technical "
        "article about current applications of multi-agent AI systems. "
        "Include an introduction, application examples, benefits, "
        "challenges, and a short source section. Keep the explanation "
        "simple and well organized."
    ),
    expected_output=(
        "A structured technical article containing an introduction, "
        "at least three applications, benefits, challenges, and source "
        "information based on the web research."
    ),
    agent=writer,
)


review_task = Task(
    description=(
        "Review the Writer's article carefully. Check accuracy, relevance, "
        "clarity, completeness, organization, and whether the article "
        "properly reflects the web research. Correct unsupported or unclear "
        "statements and produce a polished final version."
    ),
    expected_output=(
        "A polished final article that is accurate, clear, complete, "
        "well organized, and based on the web research."
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
# Run Crew
# --------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("W9D1 - CREWAI WEB SEARCH CREW")
    print("=" * 70)

    print("\nWorkflow:")
    print("Web Researcher -> Technical Writer -> Content Reviewer")
    print("\nStarting web-enabled crew...\n")

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL WEB-ENABLED CREW OUTPUT")
    print("=" * 70)
    print(result)

    print("\n" + "=" * 70)
    print("W9D1 WEB CREW EXECUTION COMPLETED")
    print("=" * 70)

    # Save output as evidence
    with open("web_output.txt", "w", encoding="utf-8") as file:
        file.write("W9D1 WEB SEARCH CREW EXECUTION EVIDENCE\n\n")
        file.write(str(result))

    print("\nOutput saved to: web_output.txt")