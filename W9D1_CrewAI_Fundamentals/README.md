\# W9D1 - CrewAI Fundamentals



\## Objective



Built a multi-agent research crew using CrewAI with three agents:

Researcher, Writer, and Reviewer.



The crew was first executed without web search and then enhanced

with a Serper web-search tool to use current web-based information.



\## Agents



\### 1. Researcher

\- Role: Researcher

\- Goal: Find accurate and relevant information

\- Responsibility: Collect and organize research findings



\### 2. Writer

\- Role: Technical Writer

\- Goal: Convert research findings into a clear explanation

\- Responsibility: Produce structured technical content



\### 3. Reviewer

\- Role: Content Reviewer

\- Goal: Check accuracy, clarity, relevance, and completeness

\- Responsibility: Improve the final response



\## Tasks



1\. Research the assigned topic.

2\. Convert research findings into a structured explanation.

3\. Review and improve the generated content.



\## Crew Process



The crew uses a sequential process:



Researcher -> Writer -> Reviewer



Each task passes its output to the next task in the workflow.



\## Local LLM



The crew uses:



\- Ollama

\- llama3.2:3b

\- Local Ollama server: http://localhost:11434



No OpenAI API key is required.



\## Basic Crew Run



The first version used the three-agent workflow without a web-search tool.



Evidence file:



basic\_output.txt



The crew successfully completed all three tasks.



\## Web-Enabled Crew



The second version added:



\- SerperDevTool

\- Real web search

\- Web-based research

\- Source information in the research workflow



Evidence file:



web\_output.txt



\## Improvement Observed



The basic crew generated an explanation using the knowledge available

to the local language model.



The web-enabled crew added current web-based information to the

Researcher's workflow. This provided additional source information

and made the research stage more suitable for topics requiring

up-to-date information.



\## Conclusion



CrewAI was used to coordinate multiple specialized agents in a

sequential workflow. Adding a web-search tool extended the

Researcher's capabilities and allowed the crew to incorporate

real web-based information before writing and reviewing the final

content.

