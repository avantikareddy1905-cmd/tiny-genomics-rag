# 🧬 Tiny Genomics RAG

This project is a lightweight, CPU-only Retrieval-Augmented Generation (RAG) pipeline designed for genomic curation.  
It reads a local set of Alzheimer’s disease text snippets and answers short questions with **inline citations** like `[S1]`.  
All components are fully open-source and run locally without GPUs or paid APIs.

---

## ⚙️ Setup Instructions (CPU-only path required)

You can run this project on any standard CPU machine.

```bash
# 1. Create a fresh environment
conda create -n ragenv python=3.10
conda activate ragenv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run RAG on a single query
python rag.py --corpus corpus.jsonl --query "What is known about APOE and Alzheimer’s disease?"

# 4. Or batch over multiple queries
python rag.py --corpus corpus.jsonl --queries queries.jsonl

retrieval settings
The program retrieves the top three sentences from the corpus that are most similar to the question being asked. Each line in the file corpus.jsonl is treated as one separate sentence or piece of information.
It measures how closely two sentences relate to each other using cosine similarity, which compares their meaning mathematically. You can also set an optional filter called --sim-threshold 0.2 so the program skips results that are weak or not related.
All sentence representations are stored locally in a small database called ChromaDB, which helps the program search and retrieve them quickly. These sentence representations, known as embeddings, are also saved as a file named embeddings.npy so that they can be reused later without needing to be recalculated, making the program run faster.

Limitations
For this project, I used a fully open and unpaid model because the goal was to keep everything local and cost-free. However, it was not as successful as OpenAI
