\# W9D4 - LangChain Hybrid Agent Systems



\## Objective



Build a hybrid LangChain system containing:



1\. A LangChain chain using PromptTemplate, Ollama LLM and StrOutputParser.

2\. ConversationBufferMemory for maintaining conversation history.

3\. A two-tool LangChain agent using a web search stub and calculator.



\## Technologies



\- Python 3.12

\- LangChain

\- LangChain Core

\- LangChain Ollama

\- LangChain Classic

\- Ollama

\- llama3.2:3b



\## Task 1 - LangChain Chain



Pipeline:



PromptTemplate

&#x20;   ↓

ChatOllama

&#x20;   ↓

StrOutputParser



The chain was tested with five different inputs.



\## Task 2 - Conversation Memory



ConversationBufferMemory was used to maintain conversation history.



Five conversation turns were tested.



The final question verified that information from previous turns could be recalled.



\## Task 3 - Two-Tool Agent



The agent was configured with two tools:



1\. Web Search Stub

2\. Calculator



Three agent tasks were executed.



\## Testing



The complete application was executed using:



```bash

python main.py

