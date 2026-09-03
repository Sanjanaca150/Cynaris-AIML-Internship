# W8D5 - Local AI Research Assistant

## Project Overview

The Local AI Research Assistant is a lightweight FastAPI-based research assistant prototype.

It uses a local knowledge base to retrieve information related to AI, Machine Learning, NLP, RAG, Large Language Models, and AI research assistants.

The project is designed to demonstrate a simple local AI research workflow without depending on external APIs.

## Objectives

- Build a local AI research assistant.
- Create a REST API using FastAPI.
- Retrieve relevant information from a local knowledge base.
- Validate API functionality using Swagger UI.
- Add automated tests using pytest.
- Follow clean coding and Git workflow practices.

## Project Structure

```text
W8D5_Local_AI_Research_Assistant
│
├── app
│   ├── __init__.py
│   └── main.py
│
├── data
│   └── knowledge_base.txt
│
├── outputs
│   ├── .gitkeep
│   └── test_results.txt
│
├── tests
│   └── test_main.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── self_review.md
```

## Technologies Used

- Python
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- HTTPX
- Local text-based knowledge base
- Git and GitHub

## How It Works

The application follows a simple workflow:

```text
User Question
      ↓
FastAPI /research endpoint
      ↓
Question validation
      ↓
Local Knowledge Base
      ↓
Keyword-based retrieval
      ↓
Research Response
```

The knowledge base is stored locally in:

```text
data/knowledge_base.txt
```

The application searches the local knowledge base for supported research topics and returns the relevant information.

## API Endpoints

### GET /

Used as a health-check endpoint.

Example response:

```json
{
  "message": "Local AI Research Assistant is running.",
  "status": "healthy"
}
```

### POST /research

Used to submit a research question.

Example request:

```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

Example response:

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "answer": "Retrieval-Augmented Generation...",
  "source": "data/knowledge_base.txt"
}
```

## Running the Application

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the FastAPI application:

```powershell
uvicorn app.main:app --reload
```

The API runs locally at:

```text
http://127.0.0.1:8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Running Tests

Run the automated test suite using:

```powershell
pytest -v
```

The project currently contains tests for:

- Health endpoint
- Valid research question
- Empty question validation
- Unknown question handling

All four tests passed successfully during development.

## Testing Evidence

Testing was performed using:

1. Swagger UI for API endpoint verification.
2. Pytest for automated testing.
3. Local knowledge-base retrieval testing.

The automated test output is stored in:

```text
outputs/test_results.txt
```

## Limitations

This prototype uses lightweight keyword-based retrieval rather than a full vector database or external Large Language Model.

The knowledge base is also intentionally small because the goal is to demonstrate the architecture and development workflow of a local research assistant.

## Future Improvements

With additional development time, the system could be improved by:

- Adding semantic/vector retrieval.
- Integrating a local LLM.
- Adding LangGraph-based workflow orchestration.
- Adding MLflow experiment and response tracking.
- Adding RAG evaluation metrics.
- Expanding the knowledge base.
- Adding authentication and production deployment.
- Adding more comprehensive API and integration tests.

## Conclusion

The project demonstrates a working local AI research assistant with a FastAPI interface, local knowledge retrieval, Swagger testing, automated pytest coverage, documentation, and a structured project layout.
