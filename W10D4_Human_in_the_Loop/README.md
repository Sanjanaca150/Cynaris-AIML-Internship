\# W10D4: Human-in-the-Loop with LangGraph



\## Objective



Build and test a three-node LangGraph workflow with conditional routing and a human-in-the-loop interrupt.



\## Graph Structure



The workflow contains three nodes:



1\. \*\*Classify\*\* – Classifies the user input into technical, greeting, support, or general.

2\. \*\*Route\*\* – Selects the route based on the classification.

3\. \*\*Respond\*\* – Requests human approval using a LangGraph interrupt before generating the final response.



\## Conditional Routing



Conditional edges route the classified input to the response node based on:



\- technical

\- greeting

\- support

\- general



\## Human-in-the-Loop



The `interrupt()` function pauses graph execution and requests human input.



The graph resumes using:



`Command(resume=human\_decision)`



`MemorySaver` is used as the checkpointing mechanism to support pause and resume.



\## Testing



Five inputs were tested:



1\. Python programming request → technical

2\. Greeting → greeting

3\. Account problem → support

4\. Artificial intelligence question → general

5\. Java coding error → technical



All five routing tests were verified successfully.



\## Human Approval Test



Input:



`I have a Python bug in my program.`



Classification:



`technical`



Route:



`technical`



Human decision:



`APPROVE`



Result:



`Human approved the technical route. Final response generated successfully.`



\## Technologies Used



\- Python 3.12

\- LangGraph

\- LangChain

\- LangChain-Ollama

\- MemorySaver

\- LangGraph interrupt

\- Command resume



\## Files



\- `human\_in\_the\_loop.py` – Main LangGraph implementation

\- `README.md` – Project documentation

