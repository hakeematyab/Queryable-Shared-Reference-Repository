<div align="center">
  <a href="https://github.com/hakeematyab/Queryable-Shared-Reference-Repository">
    <img src="https://github.com/user-attachments/assets/5192612d-bfbf-4c14-b618-1e3dbca80489" alt="Logo" width="250" height="250">
  </a>
</div>

# Queryable Shared Reference Repository - Vitek Lab

A Retrieval-Augmented Generation (RAG) system that enables lab members to search and query scientific papers through Slack, synthesizing information from multiple papers with accurate citations.

## 📋 Project Overview

**Team:** MSDS Capstone Team - 1  
**Duration:** ~10 Weeks  
**Sponsor:** Vitek Lab
**Status:** 🚧 In Development


This system will process 3,000-10,000 scientific papers and provide intelligent query capabilities through a Slack interface, deployed on NEU's Magi cluster infrastructure.

## ✨ Key Features

- **Intelligent Paper Search**: Query across thousands of research papers
- **DOI-based Paper Addition**: Add new papers in real-time using DOIs
- **Multi-Library Support**: Manage different paper collections
- **Slack Integration**: Native Slack bot for seamless lab communication
- **Hallucination Monitoring**: Quantifiable metrics for response accuracy
- **Citation Tracking**: Accurate source attribution for all responses

## 🚀 Quick Start

### Prerequisites

*To be determined*

### Installation

*To be determined*

### Configuration

*To be determined*

## 📁 Project Structure

```
qsrr/
├── data/           # Data processing and storage
├── notebooks/           # Data processing and storage
├── models/         # Model configurations and weights
├── evaluation/     # Testing and metrics
├── app/           # Slack application code
├── deployment/    # CI/CD and infrastructure
└── docs/          # Documentation
```

## 🛠️ Technical Stack

### Models
- Open-source models only
- Medium-level reasoning capability
- Local deployment on NEU infrastructure

### Infrastructure
- **Compute:** Magi cluster (M2 Ultra Mac Studios)
- **Resource Allocation:** 10-15% cluster resources
- **Users:** 1-3 concurrent (10 total max)

### Data
- **Volume:** 3,000-10,000 scientific papers
- **Formats:** PDFs, web links, .bib metadata

## 📊 Development Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1: Data & Processing** | Weeks 1-2 | PDF extraction, chunking, metadata tagging |
| **Phase 2: Modeling** | Weeks 3-5 | Embedding, generation, agent architecture |
| **Phase 3: Evaluation** | Weeks 6-7 | Metrics, testing, optimization |
| **Phase 4: Deployment** | Weeks 8-10 | Slack app, CI/CD, documentation |

## 📈 Evaluation Metrics

- **NDCG@k**: Ranking quality
- **F1 Score**: Precision/recall balance
- **ROUGE Scores**: Summary accuracy
- **RAGAS Faithfulness**: Hallucination detection
- **User Testing**: Real-world query validation

## 🧪 Testing

*To be determined*

## 🚢 Deployment

*To be determined*

## 📚 Documentation

- [User Guide](docs/user_guide.md) *(coming soon)*
- [Technical Documentation](docs/technical.md) *(coming soon)*
- [API Reference](docs/api.md) *(coming soon)*

## 🤝 Contributing

*Guidelines to be established*

## 📄 License

*To be determined*

## 👥 Team

**MSDS Capstone Team - 1**
- Atyab Hakeem - hakeem.at@northeastern.edu
- Naga Kushal Ageeru - ageeru.n@northeastern.edu
- Kishan Sathish Babu - sathishbabu.ki@northeastern.edu
- Pranav Kanth Anbarasan - anbarasan.p@northeastern.edu

**Sponsor:** Vitek Lab

## 📞 Contact

For questions about this project, please reach out to any team member listed above.

## 🙏 Acknowledgments

- Vitek Lab at Northeastern University
- MSDS Program

---

*This project is part of the MSDS Capstone requirement at Northeastern University*
