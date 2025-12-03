<div align="center">
  <a href="https://github.com/hakeematyab/Queryable-Shared-Reference-Repository">
    <img src="https://github.com/user-attachments/assets/cffe7262-d7ee-4c38-b6b2-a70612be08179" alt="Logo" width="250" height="250">
  </a>
</div>

# Queryable Shared Reference Repository

A privacy-focused, on-premises Retrieval-Augmented Generation (RAG) system that enables research groups to intelligently search and query scientific papers using natural language, with built-in hallucination detection and mitigation.

---

## 📑 Table of Contents

- [Motivation](#-motivation)

- [Objective](#-objective)

- [Key Features](#-key-features)

- [Results Highlights](#-results-highlights)

  - [Retrieval Performance](#retrieval-performance-hybrid--gte-reranking)

  - [Generation Model Performance](#generation-model-performance-qwen3-8b)

- [Hallucination Mitigation Insights](#-hallucination-mitigation-insights)

  - [Strategy 1: Confidence-Based Prompting](#strategy-1-confidence-based-prompting)

  - [Strategy 2: Context Length Management](#strategy-2-context-length-management)

- [Final Project Scorecard](#-final-project-scorecard)

- [Technical Stack](#-technical-stack)

- [Contributions](#-contributions)

- [License](#-license)

- [Acknowledgments](#-acknowledgments)

---

## 🎯 Motivation

Research groups must manage an ever-growing volume of scientific literature. While reference managers allow storage and basic retrieval, they lack intelligent, context-aware querying that integrates both paper content and metadata. Large Language Models (LLMs) can enhance search and synthesis but raise **privacy concerns** for sensitive research data and introduce risks of **hallucination** and inconsistent accuracy.

## 🎯 Objective

Develop an **on-device, shared, queryable repository** of scientific papers that:
- Enables natural language queries across thousands of papers
- Minimizes fabricated outputs through careful design and evaluation
- Ensures complete data privacy with no external API dependencies
- Operates within constrained GPU resources (~25GB VRAM)

## ✨ Key Features

- **Hybrid Retrieval-Reranking System**: Combines semantic search with BM25 lexical search with reranking for robust retrieval
- **Hallucination Detection**: Three-tiered reporting system with Bespoke RoBERTa (F1: 85.3%)
- **Hallucination Mitigation**: Confidence-based prompting achieving 93% precision and optimal context utilization findings
- **Privacy-First Design**: Fully on-premises deployment with no external API calls
- **Deployment Integration**: Agentic retrieval archiecture with friendly interface for seamless usage *(in progress)*
- **Citation Tracking**: Accurate source attribution for all responses

## 📊 Results Highlights

### Retrieval Performance (Hybrid + GTE Reranking)

| Metric | Target | Achieved |
|--------|--------|----------|
| Hit Rate@5 | ≥75% | **85.1%** |
| MRR@5 | ≥65% | **86.4%** |

<div align="left">
  <a href="https://github.com/hakeematyab/Queryable-Shared-Reference-Repository">
    <img src="https://github.com/user-attachments/assets/72d7932f-15e2-48cc-ad04-277e41cb50ee" width="720" height="512">
  </a>
</div>

### Generation Model Performance (Qwen3 8B)

| Metric | Target | Achieved |
|--------|--------|----------|
| Faithfulness | ≥85% | **88.6%** |
| Answer Relevancy | ≥80% | **80.04%** |

<div align="left">
  <a href="https://github.com/hakeematyab/Queryable-Shared-Reference-Repository">
    <img src="https://github.com/user-attachments/assets/956767be-fbb3-4379-b6e5-3ffa065a313f" width="512" height="512">
  </a>
</div>

## 🧠 Hallucination Mitigation Insights

### Strategy 1: Confidence-Based Prompting

Four prompting strategies were evaluated on Qwen3 8B:

| Strategy | Best For | Key Finding |
|----------|----------|-------------|
| **Baseline** | - | Always answers, even unanswerable queries |
| **Explicit IDK** | Clear questions | Best precision-recall tradeoff for unambiguous queries |
| **Confidence Threshold** | High-stakes | Full precision but overly conservative (20% recall) |
| **Confidence Rubric** | Ambiguous queries | Only ~6% precision drop on borderline queries vs ~29% for Explicit IDK |

<div align="left">
  <a href="https://github.com/hakeematyab/Queryable-Shared-Reference-Repository">
    <img src="https://github.com/user-attachments/assets/c1b35ed6-acff-4f98-8b97-28b5b230f0f3" width="720" height="512">
  </a>
</div>

**Recommendation**: Use **Explicit IDK** for standard queries; switch to **Confidence Rubric** when handling ambiguous or borderline questions.

### Strategy 2: Context Length Management

Investigation of "Context Rot" revealed the **"Lost in the Middle"** phenomenon:

- As context length increases, models become more conservative (fewer responses)
- Answers located in the **middle** of context are hardest to retrieve
- Answers at the **top** of context maintain better recall

<div align="left">
  <a href="https://github.com/hakeematyab/Queryable-Shared-Reference-Repository">
    <img src="https://github.com/user-attachments/assets/f3863429-9db2-42b6-a887-36ddf3883580" width="720" height="512">
  </a>
</div>

**Recommendations**:
- Limit conversations to ~10% of context window, OR
- Implement aggressive context management (summarization)
- Front-load critical information in prompts

## 📋 Final Project Scorecard

| Objective | Component | Target | Status | Result |
|-----------|-----------|--------|--------|--------|
| **Queryable Repository** | Parsing, Chunking, Embedding, Retrieval | Hit Rate@10 ≥75%, MRR@10 ≥65% | ✅ | Hit Rate@5 = 85.1%, MRR@5 = 86.4% |
| | Chat Model | Faithfulness ≥85%, Relevancy ≥80% | ✅ | Faithfulness = 88.6%, Relevancy = 80.04% |
| **Private** | GPU Memory | ≤25GB VRAM | ✅ | ~18GB VRAM |
| | Latency | Simple: <10s, Complex: <60s | ⚠️ | - |
| | External API | None | ✅ | Fully private |
| | Deployment | Architecture & Interface | ⚠️ | In Progress |
| **Groundedness** | Hallucination Detection | F1 ≥80% | ✅ | F1 = 85.3% |
| | Hallucination Mitigation | Precision ≥85% | ✅ | Precision = 93% |


## 🛠️ Technical Stack

### Selected Models
| Component | Model | Rationale |
|-----------|-------|-----------|
| Embedding | Gemma (large context) | Best Hit Rate/MRR with hybrid chunking |
| Reranker | GTE Reranker | Best MRR with larger context window for scalability |
| Retrieval | BM25 + Semantic + Reranker | Best Hit Rate, MRR for robust real-world usage |
| Generation | Qwen3 8B | Highest Faithfulness + Answer Relevancy |
| Hallucination Detection | Bespoke RoBERTa | Best F1 per billion parameters |

### Infrastructure
- **Compute:** Magi cluster (M2 Ultra Mac Studios)
- **GPU Budget:** 25GB allocation
- **Users:** 1-3 concurrent (10 total max)

### Data
- **Current:** 300 papers processed
- **Target:** 3,000-10,000 scientific papers
- **Formats:** PDFs, web links, .bib metadata


## 🚀 Quick Start
<!--
### Prerequisites

*To be determined*

### Installation

```sh
uv pip install numpy pandas scikit-learn jupyter ipykernel
uv pip install -r requirements.txt
python -m ipykernel install --user --name=QSRR --display-name "Python-QSRR" 
```

## 📁 Project Structure

```
qsrr/
├── data/           # Data processing and storage
├── notebooks/      # Experimentation notebooks
├── models/         # Model configurations and weights
├── evaluation/     # Testing and metrics
├── app/            # Slack application code
├── deployment/     # CI/CD and infrastructure
└── docs/           # Documentation
```
-->
## 📚 Documentation

- [Reports](https://github.com/hakeematyab/Queryable-Shared-Reference-Repository/tree/main/documentation)

## 🤝 Contributions

See the [GitHub Contributors Page](https://github.com/hakeematyab/Queryable-Shared-Reference-Repository/graphs/contributors) for detailed contribution history.

**Sponsor:** Vitek Lab, Northeastern University

## 📄 License

*To be determined*

## 🙏 Acknowledgments

- Vitek Lab at Northeastern University
- MSDS Program, Northeastern University

---

*This project is part of the MSDS Capstone requirement at Northeastern University*
