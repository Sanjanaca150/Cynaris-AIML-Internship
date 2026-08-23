# W7D4 — LlamaIndex + Ollama — Local RAG

## Overview

This practical implements local Large Language Model (LLM) inference using Ollama. The implementation demonstrates how to call locally hosted models through the Ollama API, use a custom system prompt, test multiple prompts, and compare the response quality of two local models.

### Models Used

* `llama3.2:3b`
* `qwen2.5:3b`

### Technologies Used

* Python
* Ollama
* Local LLM inference
* Ollama HTTP API
* JSON
* PowerShell
* Git and GitHub

---

## Practical Tasks Completed

### Task 1 — Ollama Installation and Local Inference

Ollama was installed and configured for local model execution.

The following models were used:

```text
llama3.2:3b
qwen2.5:3b
```

The first local inference was successfully tested using `llama3.2:3b`.

Example prompt:

```text
Explain what a local LLM is in two sentences.
```

The model successfully generated a response through the local Ollama API.

---

## Task 2 — Ollama API with Custom System Prompt

A Python script was created to communicate with Ollama using the local API endpoint:

```text
http://localhost:11434/api/chat
```

A custom system prompt was used:

```text
You are a helpful AI/ML internship mentor.

Answer questions clearly and accurately.
Keep explanations beginner-friendly.
For technical questions, use short examples where useful.
Do not invent facts.
```

Five prompts were tested:

1. What is an LLM?
2. What is RAG and why is it useful?
3. Explain vector embeddings in simple terms.
4. What is the difference between AI and machine learning?
5. Why can a local LLM be useful for privacy?

All five prompts successfully produced responses from the local model.

---

## Task 3 — Model Comparison

The same three questions were given to both:

* `llama3.2:3b`
* `qwen2.5:3b`

### Question 1

**Explain Retrieval-Augmented Generation (RAG) in simple terms.**

**Llama 3.2:3B:**

* Provided a more detailed technical explanation.
* Clearly identified the retriever and generator components.
* Used a story/knowledge-base example.
* Produced a relatively comprehensive response.

**Qwen 2.5:3B:**

* Used a simple library/librarian analogy.
* Was easier to understand for a beginner.
* Produced a shorter and more conversational explanation.

**Observation:** Llama 3.2 provided more technical detail, while Qwen 2.5 provided a simpler analogy.

---

### Question 2

**What is the difference between BM25 retrieval and dense retrieval?**

**Llama 3.2:3B:**

* Provided a detailed explanation of BM25 and dense retrieval.
* Discussed term frequency, ranking, vectors, and similarity.
* Produced a more technical response.

**Qwen 2.5:3B:**

* Explained BM25 using text/keyword matching.
* Explained dense retrieval using similarity between representations.
* Used simpler language and examples.

**Observation:** Llama 3.2 produced a more detailed technical response, while Qwen 2.5 was more concise and beginner-friendly.

**Quality note:** The generated responses were treated as model outputs rather than authoritative references. Some technical statements in the responses may require verification against official documentation or textbooks.

---

### Question 3

**Why might an organization choose a local LLM instead of a cloud API?**

**Llama 3.2:3B:**

* Discussed data privacy and security.
* Mentioned reduced internet dependency.
* Discussed latency, customization, cost, and compliance.
* Provided a broader list of considerations.

**Qwen 2.5:3B:**

* Focused on privacy, performance, customization, scalability, and cost.
* Used simpler explanations.
* Mentioned that cloud APIs may still be preferable for some smaller projects.

**Observation:** Llama 3.2 provided a broader and more comprehensive answer, while Qwen 2.5 gave a shorter and practical explanation.

---

## Overall Comparison

| Aspect                 | Llama 3.2:3B             | Qwen 2.5:3B                |
| ---------------------- | ------------------------ | -------------------------- |
| Response detail        | More detailed            | More concise               |
| Technical explanation  | Generally more extensive | Simpler                    |
| Beginner readability   | Good                     | Very good                  |
| Use of examples        | Good                     | Good                       |
| Response style         | Detailed and explanatory | Conversational and concise |
| Observed response time | Varied by prompt         | Varied by prompt           |

### Conclusion

Both local models successfully handled the same test questions. In these tests, `llama3.2:3b` generally produced more detailed and comprehensive explanations, while `qwen2.5:3b` generally produced simpler and more concise responses.

The comparison demonstrates that model choice depends on the application's requirements. A model that produces more detail is not automatically better for every use case.

---

## Local LLM Benefits

Running an LLM locally can provide several benefits:

* Data can remain on the local machine.
* Internet connectivity is not required for inference after the model is available.
* Users have greater control over the model and environment.
* It can reduce dependence on cloud API services.
* It can be useful for privacy-sensitive applications.

However, local LLMs also require suitable hardware, storage, memory, and local system maintenance.

---

## Project Files

```text
W7D4_LlamaIndex_Ollama/
│
├── w7d4_ollama_local_rag.py
├── output.txt
└── README.md
```

### `w7d4_ollama_local_rag.py`

Python implementation that:

* Connects to the local Ollama API.
* Uses a custom system prompt.
* Performs first local inference.
* Tests five prompts.
* Compares `llama3.2:3b` and `qwen2.5:3b`.

### `output.txt`

Contains the terminal execution output and evidence of successful completion.

### `README.md`

Contains the practical documentation, model comparison, observations, and conclusion.

---

## Execution

Run the Python script using:

```powershell
python w7d4_ollama_local_rag.py
```

Ollama must be running locally and the required models must be available.

---

## Git Workflow

The practical is maintained on the Week 7 branch:

```text
feat/aiml-W7-SanjanaCA
```

The implementation will be committed using descriptive commits and pushed to GitHub for review.

---

## Self-Review Checklist

* [x] Ollama installed and working
* [x] `llama3.2:3b` tested successfully
* [x] `qwen2.5:3b` tested successfully
* [x] Local Ollama API used
* [x] Custom system prompt implemented
* [x] Five prompts tested
* [x] Three common questions tested on both models
* [x] Model responses compared
* [x] Differences documented
* [x] Python implementation completed
* [x] Output evidence captured
* [x] README documentation added
* [ ] Minimum two Git commits completed
* [ ] Branch pushed to GitHub
* [ ] Pull Request created
* [ ] CIA Mentor Mode interaction 1 completed
* [ ] CIA Mentor Mode interaction 2 completed

---

## Viva Preparation

### 1. What is the difference between a model parameter and a prompt parameter?

A model parameter is a learned value inside the neural network, such as weights, that is adjusted during model training. A prompt parameter is information supplied by the user or application to control what the model should generate.

In simple terms, model parameters are part of the trained model, while prompt parameters are inputs provided during inference.

### 2. Why would you use a local LLM instead of a cloud API?

A local LLM can be useful when privacy, data control, offline operation, customization, or reduced dependence on external APIs is important. It allows inference to happen on the local machine instead of sending prompts and data to a remote service.

### 3. What is quantisation and how does it affect model quality?

Quantisation is the process of representing model weights using lower-precision numerical formats. It reduces memory usage and can improve inference efficiency, making it easier to run larger models on limited hardware.

The trade-off is that aggressive quantisation can reduce model quality or accuracy. The amount of quality loss depends on the quantisation method and precision used.

---

## Final Status

**W7D4 LlamaIndex + Ollama — Local RAG practical implementation completed successfully.**

The implementation demonstrates local Ollama inference, custom prompting, multi-prompt testing, and comparison of two local LLMs.
s