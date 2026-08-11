# ============================================================
# W6D2: LangChain Memory & Conversation History
# Model: Ollama qwen2.5:3b
# ============================================================

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool


# ============================================================
# SETUP
# ============================================================

# Ollama LLM for Tasks 1 and 2
llm = OllamaLLM(model="qwen2.5:3b")


# ============================================================
# TASK 1: LANGCHAIN CHAIN
# PromptTemplate -> Ollama LLM -> OutputParser
# Test with 5 inputs
# ============================================================

print("\n" + "=" * 70)
print("TASK 1: LANGCHAIN CHAIN")
print("PromptTemplate -> Ollama LLM -> OutputParser")
print("=" * 70)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Answer the following question clearly and briefly.

Question: {question}

Answer:
"""
)

parser = StrOutputParser()

chain = prompt | llm | parser

test_inputs = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is LangChain?",
    "What is an API?",
    "What is Python?"
]

for i, question in enumerate(test_inputs, start=1):
    response = chain.invoke({"question": question})

    print(f"\nInput {i}: {question}")
    print(f"Output {i}: {response}")

print("\nTASK 1 COMPLETED: 5 inputs tested.")


# ============================================================
# TASK 2: CONVERSATION HISTORY
# Maintain conversation history across 5 turns
# ============================================================

print("\n" + "=" * 70)
print("TASK 2: CONVERSATION HISTORY")
print("5 conversation turns")
print("=" * 70)

conversation_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
You are a helpful AI assistant.

Conversation history:
{history}

Current user question:
{question}

Answer the current question while using the conversation history
when it is relevant.

Answer:
"""
)

conversation_chain = conversation_prompt | llm | parser

conversation_history = []

conversation_turns = [
    "My name is Alex.",
    "I am learning Python.",
    "What programming language am I learning?",
    "What is my name?",
    "What am I currently learning?"
]

for turn_number, question in enumerate(conversation_turns, start=1):

    history_text = "\n".join(
        [
            f"User: {user_msg}\nAssistant: {assistant_msg}"
            for user_msg, assistant_msg in conversation_history
        ]
    )

    if not history_text:
        history_text = "No previous conversation."

    response = conversation_chain.invoke(
        {
            "history": history_text,
            "question": question
        }
    )

    print(f"\nTurn {turn_number}")
    print(f"User: {question}")
    print(f"Assistant: {response}")

    conversation_history.append((question, response))


print("\nStored Conversation History:")
print("-" * 70)

for i, (user_msg, assistant_msg) in enumerate(
    conversation_history, start=1
):
    print(f"Turn {i}")
    print(f"User: {user_msg}")
    print(f"Assistant: {assistant_msg}")
    print("-" * 70)

print("TASK 2 COMPLETED: Conversation history maintained across 5 turns.")


# ============================================================
# TASK 3: SIMPLE LANGCHAIN AGENT
# 2 Tools:
# 1. Web Search Stub
# 2. Calculator
# Run 3 tasks
# ============================================================

print("\n" + "=" * 70)
print("TASK 3: SIMPLE LANGCHAIN AGENT")
print("Tools: Web Search Stub + Calculator")
print("=" * 70)


# ------------------------------------------------------------
# TOOL 1: WEB SEARCH STUB
# ------------------------------------------------------------

@tool
def web_search_stub(query: str) -> str:
    """
    Simulated web search tool for learning purposes.
    This tool does not access the real internet.
    """
    return (
        f"Web search stub result for '{query}': "
        "LangChain is a framework for building applications "
        "powered by language models and connecting them with "
        "tools, prompts, memory, and other components."
    )


# ------------------------------------------------------------
# TOOL 2: CALCULATOR
# ------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Calculate a basic arithmetic expression.
    Example: 25 + 17 or 100 / 4
    """

    try:
        allowed_characters = "0123456789+-*/(). "

        if not all(
            character in allowed_characters
            for character in expression
        ):
            return "Error: Only basic arithmetic expressions are allowed."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as error:
        return f"Calculation error: {error}"


# ------------------------------------------------------------
# CREATE TOOL-CALLING CHAT MODEL
# ------------------------------------------------------------

chat_model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

tools = [
    web_search_stub,
    calculator
]


# ------------------------------------------------------------
# CREATE AGENT
# ------------------------------------------------------------

agent = create_agent(
    model=chat_model,
    tools=tools,
    system_prompt="""
You are a helpful learning assistant.

You have two tools:

1. calculator
   Use this tool for arithmetic calculations.

2. web_search_stub
   Use this tool when the user asks for web search information.

Always use the appropriate tool when necessary.
Give clear and concise final answers.
"""
)


# ------------------------------------------------------------
# RUN 3 AGENT TASKS
# ------------------------------------------------------------

agent_tasks = [
    "Calculate 25 * 4 + 10.",
    "Use the web search stub to search for information about LangChain.",
    "Calculate 144 / 12 and explain the result briefly."
]


for task_number, task in enumerate(agent_tasks, start=1):

    print(f"\nAgent Task {task_number}")
    print(f"User: {task}")

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": task
                    }
                ]
            }
        )

        final_message = result["messages"][-1]

        print(f"Agent: {final_message.content}")

    except Exception as error:

        print(f"Agent Error: {error}")


print("\nTASK 3 COMPLETED: Agent tested with 3 tasks.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("W6D2 PRACTICAL TASK SUMMARY")
print("=" * 70)

print("Task 1 - LangChain Chain: COMPLETED")
print("        5 inputs tested")

print()

print("Task 2 - Conversation History: COMPLETED")
print("        5 conversation turns tested")

print()

print("Task 3 - LangChain Agent: COMPLETED")
print("        2 tools configured")
print("        3 agent tasks tested")

print()

print("Tools used:")
print("1. Web Search Stub")
print("2. Calculator")

print()

print("W6D2 PRACTICAL IMPLEMENTATION FINISHED")
print("=" * 70)