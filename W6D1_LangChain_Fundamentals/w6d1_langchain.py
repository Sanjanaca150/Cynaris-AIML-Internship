from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent


# ============================================================
# SETUP: LOCAL OLLAMA MODEL
# ============================================================

llm = OllamaLLM(model="llama3.2:3b")
parser = StrOutputParser()


# ============================================================
# TASK 1: LANGCHAIN CHAIN
# PromptTemplate -> Ollama LLM -> OutputParser
# ============================================================

prompt = PromptTemplate(
    input_variables=["question"],
    template="""Answer the following question clearly and briefly.

Question: {question}

Answer:"""
)

chain = prompt | llm | parser

questions = [
    "What is machine learning?",
    "What is LangChain?",
    "What is a vector database?",
    "What is Python used for?",
    "What is an API?"
]

print("=" * 70)
print("W6D1 TASK 1: LANGCHAIN CHAIN")
print("=" * 70)

for i, question in enumerate(questions, start=1):
    result = chain.invoke({"question": question})

    print(f"\nInput {i}: {question}")
    print(f"Output: {result}")
    print("-" * 70)


# ============================================================
# TASK 2: CONVERSATION BUFFER MEMORY
# Verify history across 5 turns
# ============================================================

print("\n")
print("=" * 70)
print("W6D1 TASK 2: CONVERSATION BUFFER MEMORY")
print("=" * 70)

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)

conversation_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a helpful assistant.

Use the previous conversation to answer the current user message.

Previous conversation:
{history}

Current user message:
{input}

Answer:"""
)

conversation_chain = conversation_prompt | llm | parser

turns = [
    "My name is Sanjana.",
    "What is my name?",
    "I am learning LangChain.",
    "What am I learning?",
    "Can you summarize what you know about me from this conversation?"
]

for i, user_input in enumerate(turns, start=1):

    history = memory.load_memory_variables({})["history"]

    result = conversation_chain.invoke({
        "history": history,
        "input": user_input
    })

    memory.save_context(
        {"input": user_input},
        {"output": result}
    )

    print(f"\nTurn {i}")
    print(f"Human: {user_input}")
    print(f"AI: {result}")
    print("-" * 70)


print("\n")
print("STORED CONVERSATION HISTORY")
print("-" * 70)
print(memory.load_memory_variables({})["history"])


# ============================================================
# TASK 3: SIMPLE AGENT WITH 2 TOOLS
# Web Search Stub + Calculator
# ============================================================

print("\n")
print("=" * 70)
print("W6D1 TASK 3: SIMPLE AGENT")
print("=" * 70)


@tool
def web_search_stub(query: str) -> str:
    """Simulates a web search and returns information about the query."""
    return (
        f"Web search result for '{query}': "
        "LangChain is a framework for developing applications "
        "powered by language models."
    )


@tool
def calculator(expression: str) -> str:
    """Calculates a basic arithmetic expression."""

    try:
        allowed_characters = "0123456789+-*/(). "

        if not all(char in allowed_characters for char in expression):
            return "Invalid expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:
        return "Unable to calculate the expression."


tools = [
    web_search_stub,
    calculator
]


# ============================================================
# REACT AGENT PROMPT
# ============================================================

agent_prompt = PromptTemplate.from_template(
    """Answer the user's question using the available tools.

You have access to these tools:

{tools}

Use the following format exactly:

Question: the user's question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input for the tool
Observation: the result from the tool
Thought: think about the result
Final Answer: the final answer to the user

Important:
- Action must contain only the tool name.
- Do not use parentheses after the tool name.
- Put the tool input only after "Action Input:".
- Always include "Action Input:" when using a tool.

Question: {input}
Thought: {agent_scratchpad}"""
)


# ============================================================
# CREATE AGENT
# ============================================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)


# ============================================================
# RUN 3 AGENT TASKS
# ============================================================

agent_tasks = [
    "Calculate 25 * 4 + 10",
    "Use the web search tool to find information about LangChain",
    "Calculate 100 / 5 + 20"
]

for i, task in enumerate(agent_tasks, start=1):

    print(f"\nAgent Task {i}: {task}")
    print("-" * 70)

    result = agent_executor.invoke({
        "input": task
    })

    print(f"Final Answer: {result['output']}")
    print("-" * 70)


print("\n")
print("=" * 70)
print("W6D1 ALL TASKS COMPLETED")
print("=" * 70)