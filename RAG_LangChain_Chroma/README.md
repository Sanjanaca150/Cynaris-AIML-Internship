\# RAG Pipeline with LangChain \& ChromaDB



A Retrieval-Augmented Generation (RAG) question-answering chatbot built using \*\*LangChain, ChromaDB, Hugging Face embeddings, and Groq LLM\*\*.



\## Project Overview



This project implements a complete RAG pipeline that indexes a local document corpus and answers user questions using relevant retrieved information.



The system retrieves relevant document chunks from ChromaDB and provides them to a Groq language model as context. The chatbot is instructed to answer only from the indexed documents and to avoid generating unsupported information.



\## Architecture



```text

User Question

&#x20;     ↓

Question Embedding

&#x20;     ↓

ChromaDB Similarity Search

&#x20;     ↓

Relevant Document Chunks

&#x20;     ↓

Context + Question

&#x20;     ↓

Groq LLM

&#x20;     ↓

Grounded Answer

&#x20;     ↓

Source Citations

```



\## Features



\* Local document ingestion

\* Text document loading

\* Recursive text chunking

\* Hugging Face sentence embeddings

\* Persistent ChromaDB vector database

\* Semantic similarity search

\* Groq LLM integration

\* Source document citations

\* Grounded question answering

\* Unknown-question handling

\* Automated tests using pytest



\## Technologies Used



\* Python 3.12

\* LangChain

\* ChromaDB

\* LangChain Groq

\* Hugging Face Sentence Transformers

\* Groq LLM

\* python-dotenv

\* pytest



\## Project Structure



```text

RAG\_LangChain\_Chroma/

│

├── data/

│   └── documents/

│       ├── ai\_overview.txt

│       ├── machine\_learning.txt

│       └── deep\_learning.txt

│

├── app/

│   ├── \_\_init\_\_.py

│   ├── config.py

│   ├── ingest.py

│   ├── rag\_pipeline.py

│   └── chatbot.py

│

├── tests/

│   └── test\_rag.py

│

├── chroma\_db/

├── .env

├── .env.example

├── .gitignore

├── pytest.ini

├── requirements.txt

└── README.md

```



\## Setup



Create and activate a Python virtual environment:



```powershell

py -3.12 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install the required packages:



```powershell

pip install -r requirements.txt

```



\## Environment Configuration



Create a `.env` file in the project root:



```text

GROQ\_API\_KEY=your\_groq\_api\_key\_here

```



The `.env` file is excluded from Git using `.gitignore`.



\## Document Ingestion



Place `.txt` documents inside:



```text

data/documents/

```



Run the ingestion pipeline:



```powershell

python -m app.ingest

```



The documents are split into chunks, converted into embeddings, and stored in ChromaDB.



\## Run the Chatbot



Start the RAG chatbot:



```powershell

python -m app.chatbot

```



Example:



```text

You: What is machine learning?



Assistant:

Machine learning is a branch of artificial intelligence that enables

computers to learn patterns from data and make predictions or decisions

without being explicitly programmed for every individual task.



Sources:

\- machine\_learning.txt

\- ai\_overview.txt

```



\## Testing



Run the automated tests:



```powershell

pytest -q

```



The project includes tests for:



\* Document loading

\* Source metadata

\* Document splitting

\* Document directory availability



\## RAG Evaluation



The chatbot was tested with both answerable and unanswerable questions.



Example questions tested:



1\. What is machine learning?

2\. What are the three common categories of machine learning?

3\. What are transformers used for?

4\. What is the population of Mars?



The first three questions produced answers based on the indexed documents.



For the unrelated Mars question, the system correctly responded that there was not enough information in the provided documents instead of inventing an answer.



\## Source Citations



Each generated answer displays the document filenames used during retrieval.



For example:



```text

Sources:

\- deep\_learning.txt

\- ai\_overview.txt

```



These citations allow users to identify which indexed documents contributed to the response.



\## Limitations



\* The chatbot can answer only from the indexed document corpus.

\* Retrieval quality depends on chunking and embedding quality.

\* A valid Groq API key is required.

\* The demonstration corpus contains only a small number of documents.

\* Citations identify source documents rather than exact page numbers or passages.



\## Future Improvements



Possible improvements include:



\* PDF and DOCX document support

\* Streamlit web interface

\* Exact passage-level citations

\* Ragas-based evaluation

\* Conversation memory

\* Larger document collections

\* Retrieval benchmarking

\* Metadata filtering

\* Improved document preprocessing



\## Conclusion



This project demonstrates a complete Retrieval-Augmented Generation workflow using LangChain and ChromaDB with a Groq LLM. The system retrieves relevant information from a local document corpus, generates grounded answers, provides source citations, and avoids answering questions when the required information is not available in the indexed documents.



