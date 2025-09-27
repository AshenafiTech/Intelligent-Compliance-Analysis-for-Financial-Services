# Intelligent Complaint Analysis for Financial Services

## Overview
CrediTrust Financial receives thousands of customer complaints monthly across products like Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers. This project delivers an internal AI-powered tool that transforms unstructured complaint data into actionable insights using Retrieval-Augmented Generation (RAG).

**Key Features:**
- Natural language chatbot for internal teams (Product, Support, Compliance, Executives)
- Semantic search using vector database (FAISS/ChromaDB) to retrieve relevant complaint narratives
- LLM-powered answer generation with evidence from real complaints
- Multi-product querying and comparison

## Business Objectives
- **Reduce time to identify complaint trends** from days to minutes
- **Empower non-technical teams** to get answers without a data analyst
- **Enable proactive problem-solving** based on real-time feedback

## Data
- Based on Consumer Financial Protection Bureau (CFPB) complaint data
- Each record includes: issue label, free-text narrative, product/company info, submission date, and metadata

## Project Structure
```
├── data/                # Raw and processed complaint data
├── notebooks/           # EDA and prototyping notebooks
├── src/                 # Source code for API, RAG pipeline, etc.
├── tests/               # Unit and integration tests
├── requirements.txt     # Python dependencies
├── Dockerfile           # Containerization
├── docker-compose.yml   # Multi-service orchestration
└── README.md            # Project documentation
```

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AshenafiTech/Intelligent-Compliance-Analysis-for-Financial-Services.git
   cd Intelligent-Compliance-Analysis-for-Financial-Services
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Run with Docker:**
   ```bash
   docker build -t complaint-analysis .
   docker run -p 8000:8000 complaint-analysis
   ```
4. **Prepare data:**
   - Place raw complaint data in `data/raw/`
   - Process and embed data as described in the notebooks

## Usage
- To use the system, run the RAG pipeline notebooks in the `notebooks/` directory. For a chatbot or API interface, see future releases or extend using the code in `src/`.
- Ask questions like:
  - "Why are people unhappy with BNPL?"
  - "What are the top issues for Credit Cards in June 2024?"
- The system retrieves relevant complaints and generates concise, evidence-backed answers

## Contributing
1. Fork the repo and create your branch (`git checkout -b feature/your-branch-name`)
2. Commit your changes (`git commit -am 'Add new feature'`)
3. Push to the branch (`git push origin feature/your-branch-name`)
4. Open a Pull Request

## License
[MIT](LICENSE)

## Acknowledgements
- Consumer Financial Protection Bureau (CFPB) for open complaint data
- OpenAI, FAISS, ChromaDB, and other open-source contributors
