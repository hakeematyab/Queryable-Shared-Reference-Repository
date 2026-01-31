<div align="center">
  <img src="https://github.com/user-attachments/assets/cffe7262-d7ee-4c38-b6b2-a70612be08179" alt="Logo" width="200" height="200">
  
  # Queryable Shared Reference Repository
  
  **Privacy-focused, on-premises RAG system for scientific literature**
  
  [![Demo Video](https://img.shields.io/badge/▶_Watch_Demo-YouTube-red?style=for-the-badge)](https://youtu.be/rlyYthhlWJY)
  [![GitHub](https://img.shields.io/badge/View_Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/hakeematyab/Queryable-Shared-Reference-Repository)
  
  *Built for Vitek Lab, Northeastern University*
</div>

## The Problem

Research groups manage thousands of scientific papers but **can't use cloud LLMs** due to privacy concerns with sensitive research data. Existing reference managers lack intelligent querying, and LLMs **hallucinate**—fabricating citations and facts that undermine research integrity.

## The Solution

A fully **on-premises agentic RAG system** that enables natural language queries across scientific literature with built-in hallucination detection and mitigation—**no external API calls, complete data privacy**.


## Key Results

| Objective | Target | Achieved | Status |
|-----------|--------|----------|:------:|
| **Retrieval** (Hit Rate@5) | ≥75% | **85.1%** | ✅ |
| **Retrieval** (MRR@5) | ≥65% | **86.4%** | ✅ |
| **Generation Faithfulness** | ≥85% | **88.6%** | ✅ |
| **Answer Relevancy** | ≥80% | **80.04%** | ✅ |
| **Hallucination Detection** (F1) | ≥80% | **85.3%** | ✅ |
| **Hallucination Mitigation** (Precision) | ≥85% | **93%** | ✅ |
| **Latency** (Simple Query) | <10s | **~4.6s** | ✅ |
| **Latency** (Complex Query) | <60s | **~12-15s** | ✅ |
| **GPU Memory** | ≤25GB | **~18GB** | ✅ |
| **External APIs** | None | **Fully Private** | ✅ |

## Research Insights

### Hallucination Mitigation Strategies

Evaluated four prompting approaches on answerable, unanswerable, and borderline queries:

| Strategy | Best For | Precision | Recall |
|----------|----------|-----------|--------|
| Baseline | — | Low | 100% |
| **Explicit IDK** | Clear questions | ~93% | ~50% |
| Confidence Threshold | High-stakes | 100% | ~20% |
| **Confidence Rubric** | Ambiguous queries | ~87%* | ~40% |

*Only ~6% precision drop on borderline queries vs ~29% for Explicit IDK

**Recommendation:** Use Explicit IDK for standard queries; switch to Confidence Rubric for ambiguous questions.

### Context Length & "Lost in the Middle"

Discovered that model conservatism increases with context length—not hallucination rate. Key finding: **answers in the middle of long contexts are hardest to retrieve**.

**Practical guidance:**
- Limit conversations to ~10% of context window, OR
- Implement aggressive context summarization
- Front-load critical information in prompts

## Tech Stack

| Component | Selection | Rationale |
|-----------|-----------|-----------|
| **Embedding** | Gemma (8K context) | Best Hit Rate/MRR with hybrid chunking |
| **Reranker** | GTE Reranker | Best MRR + large context window for scalability |
| **Retrieval** | BM25 + Semantic + Reranker | Robust real-world performance |
| **Generation** | Qwen3 8B | Highest faithfulness + relevancy balance |
| **Hallucination Detection** | Bespoke RoBERTa | Best F1 per billion parameters |
| **Document Processing** | Docling | Layout-aware extraction with structure preservation |

**Infrastructure:** Runs on Mac Studio M2 Ultra (~18GB VRAM utilized)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/hakeematyab/Queryable-Shared-Reference-Repository.git
cd Queryable-Shared-Reference-Repository/app
chmod +x setup.sh startup.sh shutdown.sh

# Start all services
./startup.sh

# Access at http://localhost:3000
```

**Requirements:** Python 3.11+, Node.js 18+, Ollama, ~25GB VRAM

## Features

- **Natural Language Queries** — Ask questions across your paper collection
- **Three-Tiered Trust Badges** — Visual grounding indicators (✓ Green, ⚠ Amber, ✕ Red)
- **Citation Tracking** — Source attribution for all responses
- **Paper Ingestion** — Upload PDFs directly through the interface
- **Chat History** — Persistent conversations per user
- **Fully Private** — No external API calls, runs entirely on-premises


## Documentation

- [Technical Report](https://github.com/hakeematyab/Queryable-Shared-Reference-Repository/blob/main/documentation/DSCapstoneP2Report.pdf)
