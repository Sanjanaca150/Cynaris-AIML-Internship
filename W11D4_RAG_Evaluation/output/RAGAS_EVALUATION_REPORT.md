\# W11D4 RAG Evaluation Report



\## Project



W11D4: RAG Evaluation with Ragas — Metrics \& Benchmarks



\## Dataset



Healthcare knowledge dataset containing 10 topics:



1\. Diabetes

2\. Hypertension

3\. Asthma

4\. Anemia

5\. Common Cold

6\. Influenza

7\. Heart Disease

8\. Kidney Health

9\. Nutrition

10\. Preventive Healthcare



\## Evaluation Dataset



10 question-answer pairs were generated from the healthcare knowledge dataset and stored in:



`output/evaluation\_dataset.json`



\## Baseline RAG Configuration



\- Chunk size: 300

\- Chunk overlap: 50

\- Retrieval k: 1

\- Embedding model: `nomic-embed-text:latest`

\- Generation model: `llama3.2:3b`

\- Vector store: ChromaDB



\## Optimized RAG Configuration



The retrieval configuration was optimized by increasing the number of retrieved chunks:



\- Chunk size: 300

\- Chunk overlap: 50

\- Retrieval k: 3

\- Embedding model: `nomic-embed-text:latest`

\- Generation model: `llama3.2:3b`

\- Vector store: ChromaDB



The optimized pipeline successfully retrieved 3 chunks for each query and generated context-based answers.



\## Ragas Metrics



The following Ragas metrics were attempted:



\- Faithfulness

\- Answer Relevancy

\- Context Precision

\- Context Recall



\## Ragas Evaluation Attempt



Ragas 0.4.3 was used with a local Ollama model as the evaluation judge.



The evaluation was attempted using:



\- `llama3.2:3b`

\- `qwen2.5:3b`



Both local models produced timeout errors and invalid structured JSON responses during Ragas metric evaluation.



A single-question diagnostic test was also performed using Faithfulness. The evaluation timed out after approximately 3 minutes and returned:



`{'faithfulness': nan}`



The full 10-question optimized evaluation completed its 40 metric jobs, but the local judge generated multiple `OutputParserException` and `TimeoutError` failures. Therefore, valid numerical scores could not be reliably produced for the complete evaluation.



\## Important Evaluation Note



No Ragas scores were fabricated or manually assigned.



The `null` values in the generated JSON result files indicate that the local Ragas judge evaluation did not produce reliable numerical results.



This limitation is specific to the local LLM-as-judge evaluation setup. The RAG retrieval and answer-generation pipeline itself executed successfully.



\## Optimization Performed



The retrieval parameter was changed from:



`k = 1`



to:



`k = 3`



The optimized pipeline was rebuilt with a clean ChromaDB collection to avoid duplicate vector entries.



The optimized RAG pipeline successfully retrieved 3 context chunks and generated answers from the retrieved healthcare information.



\## Baseline Ragas Console Evidence



During the initial baseline evaluation, Ragas displayed the following aggregate values:



\- Faithfulness: 1.0000

\- Answer Relevancy: 0.8528

\- Context Precision: 0.0000

\- Context Recall: 0.9286



These values were displayed by the Ragas evaluation process, but the original result-extraction code did not preserve them correctly in the JSON output. Because the complete evaluation also contained failed jobs, these values are retained as console evidence rather than treated as a complete reliable benchmark.



\## Conclusion



The W11D4 RAG evaluation workflow was implemented successfully, including dataset preparation, baseline retrieval, optimized retrieval, ChromaDB indexing, Ollama generation, and Ragas evaluation attempts.



The main limitation encountered was the inability of the available local 3B Ollama models to consistently satisfy Ragas' structured LLM-judge output requirements within the available execution time.



The optimization experiment was implemented by increasing retrieval k from 1 to 3, and the optimized RAG pipeline was verified independently.

