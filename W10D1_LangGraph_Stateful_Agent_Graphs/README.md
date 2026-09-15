\# W10D1: LangGraph — Stateful Agent Graphs



\## Objective



Built a three-node LangGraph stateful agent with classification, conditional routing, response generation, and human-in-the-loop interruption.



\## Graph Architecture



The main graph follows:



`START → classify → route → respond → END`



\### Nodes



1\. \*\*classify\*\*



&#x20;  \* Classifies user input into:



&#x20;    \* technical

&#x20;    \* career

&#x20;    \* general



2\. \*\*route\*\*



&#x20;  \* Determines the routing path based on the classification result.



3\. \*\*respond\*\*



&#x20;  \* Generates a response according to the selected route.



\## Conditional Routing



Conditional edges are used after the `route` node to direct execution based on the classification output.



Supported routes:



\* technical

\* career

\* general



\## Testing



Five inputs were tested:



1\. `I want to learn Python` → technical

2\. `I have a Java programming question` → technical

3\. `How should I prepare for an interview?` → career

4\. `Hello, good morning` → general

5\. `I need help with SQL` → technical



All five inputs were classified and routed correctly.



\## Human-in-the-Loop



A human review node was added using LangGraph's `interrupt()` mechanism.



The workflow:



`classify → route → respond → human\_review`



The graph pauses during human review and waits for human input.



Tested workflow:



`Pause → Human input: APPROVE → Resume`



The graph successfully resumed and produced the final approved response.



\## Technologies Used



\* Python 3.12

\* LangGraph

\* LangChain

\* LangChain-Ollama

\* LangGraph MemorySaver



\## Result



The W10D1 practical requirements were successfully implemented and tested.



