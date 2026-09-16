from crewai import Agent, Crew, LLM, Task


# Configure CrewAI to use the local Ollama model.
# This avoids requiring an OPENAI_API_KEY.
ollama_llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0,
)


def create_support_crew(customer_name: str, issue: str):
    """Create a CrewAI customer support analysis workflow."""

    support_agent = Agent(
        role="Customer Support Specialist",
        goal=(
            "Analyze customer issues and provide accurate, "
            "professional support recommendations."
        ),
        backstory=(
            "An experienced customer support specialist who "
            "focuses on clear and customer-friendly communication."
        ),
        llm=ollama_llm,
        verbose=False,
        allow_delegation=False,
    )

    support_task = Task(
        description=(
            f"Analyze the following issue for customer "
            f"{customer_name}: {issue}. "
            "Provide a short recommendation for the appropriate "
            "customer support action."
        ),
        expected_output=(
            "A concise support recommendation and next action."
        ),
        agent=support_agent,
    )

    return Crew(
        agents=[support_agent],
        tasks=[support_task],
        verbose=False,
    )