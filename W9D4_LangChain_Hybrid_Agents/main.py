from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory
from langchain.agents import create_agent
from langchain.tools import tool


# ============================================================
# W9D4 - CrewAI + LangChain: Hybrid Agent Systems
# ============================================================

MODEL_NAME = "llama3.2:3b"


# ------------------------------------------------------------
# Shared Ollama LLM
# ------------------------------------------------------------

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


# ============================================================
# TASK 1
# PromptTemplate -> Ollama LLM -> StrOutputParser
# ============================================================

def run_chain_demo():
    print("\n" + "=" * 70)
    print("TASK 1: LANGCHAIN CHAIN")
    print("=" * 70)

    prompt = PromptTemplate.from_template(
        """
        You are a concise AI assistant.

        Answer the following question in 2-3 simple sentences.

        Question: {question}

        Answer:
        """
    )

    chain = prompt | llm | StrOutputParser()

    test_inputs = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is LangChain?",
        "What is an AI agent?",
        "Why is Python useful for AI?"
    ]

    for index, question in enumerate(test_inputs, start=1):
        print(f"\nTest {index}")
        print(f"Input: {question}")

        response = chain.invoke({
            "question": question
        })

        print(f"Output: {response}")


# ============================================================
# TASK 2
# ConversationBufferMemory
# ============================================================

def run_memory_demo():
    print("\n" + "=" * 70)
    print("TASK 2: CONVERSATION BUFFER MEMORY")
    print("=" * 70)

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=False
    )

    conversation_prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant.

        Previous conversation:
        {history}

        Current user message:
        {input}

        Answer naturally and briefly.
        """
    )

    conversation_chain = conversation_prompt | llm | StrOutputParser()

    turns = [
        "My name is Sanjana.",
        "I am an Information Science student.",
        "I am learning artificial intelligence.",
        "I am currently working on a LangChain task.",
        "What is my name and what am I currently learning?"
    ]

    for index, user_input in enumerate(turns, start=1):
        print(f"\nTurn {index}")
        print(f"User: {user_input}")

        history = memory.load_memory_variables({})["history"]

        response = conversation_chain.invoke({
            "history": history,
            "input": user_input
        })

        print(f"Assistant: {response}")

        memory.save_context(
            {"input": user_input},
            {"output": response}
        )

    print("\n--- FINAL STORED CONVERSATION HISTORY ---")
    print(memory.load_memory_variables({})["history"])


# ============================================================
# TASK 3
# Two-tool LangChain Agent
# ============================================================

@tool
def web_search_stub(query: str) -> str:
    """
    Simulates a web search and returns a small set of
    predefined research results. This is a stub and does
    not access the live internet.
    """

    query_lower = query.lower()

    results = {
        "langchain": (
            "LangChain is a framework for building applications "
            "with language models, tools, agents and memory."
        ),
        "crewai": (
            "CrewAI is a framework for orchestrating role-based "
            "AI agents that collaborate on tasks."
        ),
        "ollama": (
            "Ollama allows local execution of open-source language "
            "models on a computer."
        ),
        "python": (
            "Python is widely used in AI and machine learning "
            "because of its simple syntax and rich ecosystem."
        )
    }

    for keyword, result in results.items():
        if keyword in query_lower:
            return f"Search result for '{query}': {result}"

    return (
        f"Search result for '{query}': "
        "No predefined result was found. "
        "This is a demonstration web-search stub."
    )


@tool
def calculator(expression: str) -> str:
    """
    Calculates a basic mathematical expression.
    Supports numbers and standard arithmetic operators.
    """

    allowed_characters = "0123456789+-*/(). "

    if not all(char in allowed_characters for char in expression):
        return "Calculator error: unsupported characters."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation result: {result}"
    except Exception as error:
        return f"Calculator error: {error}"


def run_agent_demo():
    print("\n" + "=" * 70)
    print("TASK 3: TWO-TOOL LANGCHAIN AGENT")
    print("=" * 70)

    tools = [
        web_search_stub,
        calculator
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful research assistant. "
            "Use the web_search_stub tool for research questions. "
            "Use the calculator tool for mathematical calculations. "
            "When a task requires calculation, use the calculator tool. "
            "Give a concise final answer."
        )
    )

    tasks = [
        "Search for information about LangChain.",
        "Calculate 125 * 8 + 50.",
        "Search for information about Ollama and explain it briefly."
    ]

    for index, task in enumerate(tasks, start=1):
        print(f"\nAgent Task {index}")
        print(f"User: {task}")

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": task
                }
            ]
        })

        final_message = result["messages"][-1]

        print(f"Agent: {final_message.content}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("#" * 70)
    print("W9D4 - LANGCHAIN HYBRID AGENT SYSTEMS")
    print("#" * 70)

    run_chain_demo()

    run_memory_demo()

    run_agent_demo()

    print("\n" + "#" * 70)
    print("W9D4 COMPLETED")
    print("#" * 70)