# Fine-Tuning and RAG for a Course Exam-Prep Chatbot

This project compares two different ways to turn a small open-source LLM into a chatbot that can answer questions about a specific course (CS 639, Data Management for Data Science, at UW-Madison): fine-tuning the model directly on course material, and Retrieval-Augmented Generation (RAG), which instead retrieves relevant content at query time and hands it to the model as context.

The notebook runs through both approaches end to end and finishes with a head-to-head comparison of the two. First, a 4-bit quantized `Llama-3.2-1B-Instruct` model is tested on its own, then fine-tuned with LoRA on question-answer pairs synthesized from 23 CS 639 lecture transcripts using a larger teacher model (Qwen2.5-7B-Instruct). Separately, a RAG pipeline is built with Elasticsearch and Haystack, indexing the same transcripts and retrieving relevant chunks to answer questions through a Streamlit chat app. Along the way, the notebook also compares two different chunking strategies for the RAG index (fixed-size token windows vs. sentence-grouped chunks) and measures which one retrieves more relevant results.

## Technologies

| Category | Tools |
|---|---|
| Model | `Llama-3.2-1B-Instruct` (4-bit quantized), `Qwen2.5-7B-Instruct` (teacher model for QA generation) |
| Fine-tuning | LoRA (via `peft`), `trl`'s `SFTTrainer`, `bitsandbytes` |
| Retrieval | Elasticsearch, Haystack (BM25 retrieval, RAG pipeline) |
| Generation | HuggingFace Inference API |
| App | Streamlit, deployed locally via localtunnel during development |
| Environment | Google Colab (T4 GPU for Sections 1 & 2) |

## Repo structure

| File | Description |
|---|---|
| `fine_tuning_and_rag_chatbot.ipynb` | The full notebook: fine-tuning, RAG, and the comparison |
| `images/streamlit_chatbot.png` | Screenshot of the deployed chat app |
| `requirements.txt` | Python dependencies |

## Data

The notebook is built around 23 `.txt` transcripts of CS 639 lectures. Those transcripts aren't included in this repo, since they're transcribed course lecture content rather than something I'm able to redistribute. The code that cleans, chunks, and indexes them is all here, and would work on any similarly formatted set of transcripts (or with light changes, any text corpus split into per-document `.txt` files).

The 925 synthesized question-answer pairs used for fine-tuning, generated from those transcripts by the teacher model, aren't included either, for the same reason: they're derived directly from the lecture content.

## Sections

**Section 1 — Text generation with a pre-trained LLM.** Loads the quantized Llama model, tests it on a few prompts, looks at a case where it fails (asking for information the model can't know, like the exact contents of an exam), and tries a chat-template system prompt to see if the model adopts an assigned role.

**Section 2 — Fine-tuning on lecture transcripts.** Cleans and chunks the transcripts, uses Qwen2.5-7B-Instruct to synthesize exam-style QA pairs from each chunk, and fine-tunes the Llama model on those pairs with LoRA. The fine-tune is evaluated two ways: qualitatively, by comparing the base and fine-tuned model's answers to the same course questions, and quantitatively, by comparing perplexity on a held-out test set.

**Section 3 — A RAG-based exam-prep chatbot.** Indexes the transcripts into Elasticsearch two different ways (fixed-size token windows and sentence-grouped chunks), compares retrieval precision between them, then builds a RAG pipeline with Haystack and deploys it as a Streamlit chat app. The notebook closes with a direct comparison of the fine-tuned model against RAG on the same set of test prompts.

## Environment

Sections 1 and 2 need a GPU (a free Colab T4 session is enough) and a HuggingFace account with access to the gated `Llama-3.2-1B-Instruct` model. Section 3 doesn't need a GPU, but does need an Elastic Cloud account and a HuggingFace API token, since generation goes through HuggingFace's hosted inference API rather than a locally loaded model.

Since the lecture transcripts aren't included, this repo isn't meant to be cloned and run end to end — it's meant to be read, with the code and its original results both visible directly in the notebook.

```bash
pip install -r requirements.txt
```

## Key insights and learnings

Sentence-level chunking retrieved more precisely than fixed-size token windows for this data (Precision@3 of 0.87 vs. 0.53 across 5 test questions), likely because keeping each chunk to a complete set of sentences avoids splitting a relevant idea across two chunks the way a fixed token window can.

RAG also outperformed fine-tuning on the head-to-head comparison. The fine-tuned model tended to give shorter, more generic answers and occasionally hallucinated details (for example, describing SQL window functions using the wrong clause), while RAG's answers were grounded in the actual retrieved transcript content and held up better on questions that weren't directly covered in training.
