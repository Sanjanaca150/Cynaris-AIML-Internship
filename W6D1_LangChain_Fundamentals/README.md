# W6D1: LangChain Fundamentals — Chains & Prompts

## Objective

This project demonstrates the fundamentals of LangChain using a local Ollama LLM.

## Practical Tasks

### Task 1: LangChain Chain

Built a LangChain chain using:

PromptTemplate → Ollama LLM → StrOutputParser

The chain was tested with 5 different inputs.

### Task 2: ConversationBufferMemory

Added ConversationBufferMemory to maintain conversation history.

The conversation was tested across 5 turns to verify that information from previous turns could be accessed.

### Task 3: Simple Agent

Built a simple LangChain ReAct agent with two tools:

1. Web Search Stub
2. Calculator

The agent was tested with 3 tasks.

## Technologies Used

- Python
- LangChain
- Ollama
- llama3.2:3b
- ConversationBufferMemory
- ReAct Agent

## How to Run

Make sure Ollama is running and the `llama3.2:3b` model is available.

Install the required packages:

```bash
pip install -r requirements.txt