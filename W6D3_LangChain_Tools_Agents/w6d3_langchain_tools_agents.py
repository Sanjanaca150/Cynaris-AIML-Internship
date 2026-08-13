# ============================================================
# W6D3: LangChain Tools & Agents
# Complete Practical Tasks - Single Program
# ============================================================

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent


# ============================================================
# SETUP
# ============================================================

MODEL_NAME = "llama3.2:3b"

llm = OllamaLLM(model=MODEL_NAME)
parser = StrOutputParser()


# ============================================================
# TASK 1
# CHAIN: PromptTemplate -> Ollama LLM -> OutputParser
# TEST WITH 5 INPUTS
# ============================================================

print("\n" + "=" * 70)
print("TASK 1: LANGCHAIN CHAIN")
print("=" * 70)

prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Explain {topic} in simple terms. "
        "Give a short answer in 2-3 sentences."
    )
)

chain = prompt | llm | parser

topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "LangChain",
    "Python",
    "Cloud Computing"
]

for number, topic in enumerate(topics, start=1):
    result = chain.invoke({"topic": topic})

    print(f"\nInput {number}: {topic}")
    print(f"Output: {result}")

print("\nTASK 1 STATUS: COMPLETED")
print("Chain tested successfully with 5 inputs.")


# ============================================================
# TASK 2
# CONVERSATION MEMORY / HISTORY
# VERIFY 5 TURNS
# ============================================================

print("\n" + "=" * 70)
print("TASK 2: CONVERSATION MEMORY / HISTORY")
print("=" * 70)

conversation_history = []

memory_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
You are a helpful assistant.

Previous conversation:
{history}

Current user question:
{question}

Use the previous conversation to answer the current question.
Keep your answer short and clear.
"""
)

memory_chain = memory_prompt | llm | parser

conversation_turns = [
    "My name is Sanjana.",
    "What is my name?",
    "I am learning LangChain.",
    "What am I learning?",
    "What do you know about me from this conversation?"
]

for turn_number, question in enumerate(conversation_turns, start=1):

    history_text = ""

    for message in conversation_history:
        history_text += (
            f"{message.type}: {message.content}\n"
        )

    response = memory_chain.invoke({
        "history": history_text,
        "question": question
    })

    print(f"\nTurn {turn_number}")
    print(f"User: {question}")
    print(f"Assistant: {response}")

    conversation_history.append(
        HumanMessage(content=question)
    )

    conversation_history.append(
        AIMessage(content=response)
    )

print("\nTASK 2 STATUS: COMPLETED")
print("Conversation history maintained across 5 turns.")


# ============================================================
# TASK 3
# LANGCHAIN AGENT
# TWO TOOLS:
# 1. Web Search Stub
# 2. Calculator
# TEST WITH 3 TASKS
# ============================================================

print("\n" + "=" * 70)
print("TASK 3: LANGCHAIN AGENT")
print("=" * 70)


# ------------------------------------------------------------
# TOOL 1: WEB SEARCH STUB
# ------------------------------------------------------------

@tool
def web_search_stub(query: str) -> str:
    """Simulates a web search and returns a search result."""
    return (
        f"Web Search Stub Result: "
        f"Information found for '{query}'."
    )


# ------------------------------------------------------------
# TOOL 2: CALCULATOR
# ------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""

    try:
        allowed_characters = (
            "0123456789+-*/(). "
        )

        if not all(
            character in allowed_characters
            for character in expression
        ):
            return "Invalid mathematical expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:
        return "Unable to calculate the expression."


# ------------------------------------------------------------
# CREATE CHAT MODEL
# ------------------------------------------------------------

agent_llm = ChatOllama(
    model=MODEL_NAME
)


# ------------------------------------------------------------
# CREATE AGENT
# ------------------------------------------------------------

agent = create_agent(
    model=agent_llm,
    tools=[
        web_search_stub,
        calculator
    ],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the calculator tool for mathematical calculations. "
        "Use the web_search_stub tool for web search requests."
    )
)


# ------------------------------------------------------------
# THREE AGENT TASKS
# ------------------------------------------------------------

agent_tasks = [
    "Calculate 25 * 16.",
    "Search the web for information about LangChain.",
    "Calculate (100 + 50) / 5."
]

for task_number, task in enumerate(agent_tasks, start=1):

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": task
            }
        ]
    })

    final_answer = result["messages"][-1].content

    print(f"\nAgent Task {task_number}")
    print(f"User: {task}")
    print(f"Agent: {final_answer}")

print("\nTASK 3 STATUS: COMPLETED")
print("Agent tested successfully with 3 tasks.")
print("Agent contains 2 tools: Web Search Stub + Calculator.")


# ============================================================
# FINAL W6D3 SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("W6D3 PRACTICAL TASK SUMMARY")
print("=" * 70)

print("""
[COMPLETED] Task 1
PromptTemplate -> Ollama LLM -> OutputParser
Tested with 5 inputs

[COMPLETED] Task 2sss
Conversation history / memory
Verified across 5 conversation turns

[COMPLETED] Task 3
LangChain Agent
Two tools:
  1. Web Search Stub
  2. Calculator
Tested with 3 tasks

W6D3 PRACTICAL TASKS: COMPLETED
""")

print("=" * 70)
print("END OF W6D3 PRACTICAL EXECUTION")
print("=" * 70)