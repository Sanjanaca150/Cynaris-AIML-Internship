import os
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

# 1. Initialize search tool
search_tool = SerperDevTool()

# 2. Define Researcher Agent with search capabilities
researcher = Agent(
    role="Senior AI Researcher",
    goal="Gather factual, up-to-date information about the CrewAI framework.",
    backstory="You are an expert technical researcher who verifies information on the web before writing.",
    tools=[search_tool],
    verbose=True
)

# 3. Define the Research Task
research_task = Task(
    description="Search the web and provide an accurate summary of CrewAI (Python LLM Agent Framework).",
    expected_output="A structured summary explaining CrewAI's core concepts: Agents, Tasks, Tools, and Crews.",
    agent=researcher
)

# 4. Form and run the Crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential
)

# 5. Execute
output = crew.kickoff()
print("\n=== FINAL OUTPUT ===")
print(output)