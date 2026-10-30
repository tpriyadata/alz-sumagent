---
title: Alz Research Assistant
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Alz Research Assistant

A RAG-powered chatbot that answers Alzheimer's disease research questions
grounded in 19,600 PubMed papers. Designed for researchers, clinicians,
and students seeking evidence-based insights.

## Architecture

```
User question
      ↓
PubMedBert embeddings (local)
      ↓
Pinecone vector search (top_k=20)
      ↓
Cross-encoder reranking (top 5)
      ↓
Claude Sonnet generation
      ↓
Grounded answer with PMID citations
```

## Evaluation

`python -c "from evaluation.ragas_eval import run_evaluation; run_evaluation()"`
runs 10 gold questions plus 10 adversarial cases (personal medical advice, prompt
injection, out-of-corpus questions, false-positive guards) and writes
`evaluation/report.json` with pass rates per category.

What the metrics mean:

- **hit rate**: retrieval recall against hand-labelled PMIDs
- **keyword coverage**: crude answer-relevancy proxy
- **citation validity**: structural check that cited PMIDs were retrieved. It does
  **not** verify that a claim is supported by the source text, so it is not
  faithfulness. Claim-level faithfulness (LLM judge calibrated on human labels)
  is not implemented yet.

Errored questions count as failures. Scores are not hard-coded here; run the
evaluation to produce them.

## Guardrails

- **Entry gate** (`generation/guards.py`): deterministic; declines personal medical
  advice and prompt-injection attempts before any retrieval or LLM call.
- **Prompt hygiene**: retrieved text is wrapped in delimiters and treated as untrusted
  data; one versioned prompt (`PROMPT_VERSION`) is recorded on every answer.
- **Exit gate** (`validate_citations`): every cited PMID must belong to the retrieved
  sources. Unverifiable findings are removed and flagged; a summary citing an
  unretrieved PMID, or unparseable model output, is withheld (fails closed).

## Tests and CI

```
pip install -r requirements-ci.txt
pytest
```

Unit tests mock the vector store and the LLM, so they need no API keys. GitHub
Actions runs them on every push and pull request.

## Summarizer model

[oviya08/Alz-SumAgent](https://huggingface.co/oviya08/Alz-SumAgent)
— BART-large for PubMed abstract summarization.

## Stack

- **LLM**: Claude Sonnet (Anthropic)
- **Vector DB**: Pinecone
- **Embeddings**: pritamdeka/S-PubMedBert-MS-MARCO
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Summarizer**: oviya08/Alz-SumAgent
- **Frontend**: Streamlit

## Space secrets required

Set in HuggingFace Space Settings → Secrets:

```
ANTHROPIC_API_KEY
PINECONE_API_KEY
PUBMED_EMAIL
HF_TOKEN
```