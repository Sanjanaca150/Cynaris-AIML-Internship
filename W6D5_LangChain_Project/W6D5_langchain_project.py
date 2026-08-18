"""
W6D5 - Week 6 Project: Document Chatbot with LangChain

Tasks:
1. PromptTemplate -> Ollama LLM -> StrOutputParser (5 inputs)
2. ConversationBufferMemory across 5 turns
3. LangChain Agent with 2 tools:
   - Web search stub
   - Calculator

Required stack:
LangChain + Ollama
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain.agents import create_agent

import ast
import operator as op


# ================================================================
# CONFIGURATION
# ================================================================

MODEL = "llama3.2:3b"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# TASK 1: LANGCHAIN CHAIN
# PromptTemplate -> Ollama -> StrOutputParser
# ================================================================

section("TASK 1: LANGCHAIN CHAIN")

# Ollama LLM for normal LangChain chain
llm = OllamaLLM(model=MODEL)

prompt = PromptTemplate.from_template(
    """You are a knowledgeable technical assistant.

Answer the question accurately.
Use simple language.
Do not invent information.
If the question is about LangChain, LLMs, prompt templates,
vector databases, or RAG, give the standard technical definition.

Answer in 2-4 sentences.

Question: {question}

Answer:"""
)

parser = StrOutputParser()

chain = prompt | llm | parser

inputs = [
    "What is LangChain?",
    "What is an LLM?",
    "What is a prompt template?",
    "What is a vector database?",
    "What is RAG?"
]

for i, question in enumerate(inputs, 1):
    print(f"\nInput {i}: {question}")

    try:
        answer = chain.invoke({"question": question})
        print(f"Output {i}: {answer}")

    except Exception as e:
        print(f"Output {i}: ERROR - {e}")


# ================================================================
# TASK 2: CONVERSATION BUFFER MEMORY
# ================================================================

section("TASK 2: CONVERSATION BUFFER MEMORY")

memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    output_key="response",
    return_messages=False
)

conversation_prompt = PromptTemplate.from_template(
    """You are a helpful conversational assistant.

Use the conversation history to answer the user's latest message.
Remember information that the user provided earlier.

Conversation history:
{history}

User: {input}

Assistant:"""
)

conversation_chain = conversation_prompt | llm | parser

turns = [
    "My name is Sanjana.",
    "I am learning LangChain during my internship.",
    "What topic am I learning?",
    "What is my name?",
    "Can you summarize what you know about this conversation?"
]

for i, user_input in enumerate(turns, 1):

    print(f"\nTurn {i}")
    print(f"User: {user_input}")

    try:
        history = memory.load_memory_variables({}).get("history", "")

        response = conversation_chain.invoke(
            {
                "history": history,
                "input": user_input
            }
        )

        print(f"Assistant: {response}")

        memory.save_context(
            {"input": user_input},
            {"response": response}
        )

    except Exception as e:
        print(f"Assistant: ERROR - {e}")


print("\nFINAL STORED CONVERSATION HISTORY:")

print(
    memory.load_memory_variables({}).get("history", "")
)


# ================================================================
# TASK 3: LANGCHAIN AGENT WITH 2 TOOLS
# ================================================================

section("TASK 3: LANGCHAIN AGENT WITH 2 TOOLS")


# ----------------------------------------------------------------
# TOOL 1: WEB SEARCH STUB
# ----------------------------------------------------------------

@tool
def web_search_stub(query: str) -> str:
    """
    Simulated web search tool for the internship task.
    This tool does not access the real internet.
    """

    query_lower = query.lower()

    if "langchain" in query_lower:
        return (
            "LangChain is an open-source framework for developing "
            "applications powered by large language models. "
            "It provides components for prompts, chains, agents, "
            "retrieval, memory, and tool use."
        )

    if "ollama" in query_lower:
        return (
            "Ollama is a local AI model runtime that allows users "
            "to run supported large language models locally on "
            "their own computer."
        )

    return (
        f"Search result for '{query}': "
        "This is simulated web-search information "
        "created for the internship task."
    )


# ----------------------------------------------------------------
# TOOL 2: SAFE CALCULATOR
# ----------------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node):

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

    if isinstance(node, ast.UnaryOp):
        if type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](
                _safe_eval(node.operand)
            )

    if isinstance(node, ast.BinOp):
        if type(node.op) in _ALLOWED_OPERATORS:

            left = _safe_eval(node.left)
            right = _safe_eval(node.right)

            return _ALLOWED_OPERATORS[type(node.op)](
                left,
                right
            )

    raise ValueError(
        "Only basic arithmetic expressions are allowed."
    )


@tool
def calculator(expression: str) -> str:
    """
    Calculate a basic arithmetic expression.
    Example: 25*4+10
    """

    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

        result = _safe_eval(tree)

        return str(result)

    except Exception as e:

        return f"Calculator error: {e}"


# ================================================================
# IMPORTANT:
# ChatOllama is used for the agent because it supports
# tool calling / bind_tools.
# ================================================================

agent_llm = ChatOllama(
    model=MODEL,
    temperature=0
)


# ================================================================
# CREATE LANGCHAIN AGENT
# ================================================================

agent = create_agent(
    model=agent_llm,
    tools=[
        web_search_stub,
        calculator
    ],
    system_prompt=(
        "You are a helpful internship assistant. "

        "You have two tools. "

        "Tool 1: web_search_stub. "
        "Use it for questions asking for web/search information. "

        "Tool 2: calculator. "
        "Use it for arithmetic calculations. "

        "When a user asks for both search information "
        "and a calculation, use both tools. "

        "Always give the final answer after using the required tools."
    )
)


# ================================================================
# AGENT TEST TASKS
# ================================================================

agent_tasks = [

    "Use the web search tool to tell me what LangChain is.",

    "Calculate 125 * 8 + 50.",

    (
        "Use the web search tool to find the stub information "
        "about Ollama, then calculate 45 / 5 and include both "
        "results in your answer."
    )
]


# ================================================================
# RUN AGENT TASKS
# ================================================================

for i, task in enumerate(agent_tasks, 1):

    print(f"\nAgent Task {i}: {task}")

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

        messages = result.get("messages", [])

        # Find the final AI message
        final_message = None

        for message in reversed(messages):

            if getattr(message, "type", "") == "ai":

                content = getattr(
                    message,
                    "content",
                    ""
                )

                if content:
                    final_message = message
                    break

        if final_message is not None:

            print(
                f"Agent Output {i}: "
                f"{final_message.content}"
            )

        else:

            print(
                f"Agent Output {i}: "
                "No final response generated."
            )

    except Exception as e:

        print(
            f"Agent Output {i}: ERROR - {e}"
        )


# ================================================================
# COMPLETION SUMMARY
# ================================================================

section("W6D5 COMPLETION SUMMARY")

print("Task 1: Chain created and tested with 5 inputs.")
print("Task 2: ConversationBufferMemory tested across 5 turns.")
print("Task 3: Agent created with 2 tools and tested with 3 tasks.")
print("Required stack: LangChain + Ollama.")
print("Output evidence is printed in the terminal.")
print("W6D5 practical implementation completed.")