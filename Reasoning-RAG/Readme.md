# 🧪 CNT Research Assistant

The **CNT Research Assistant** is an advanced, interactive application designed to answer complex questions about **Carbon Nanotubes (CNTs)**. It uses a **Retrieval-Augmented Generation (RAG)** architecture to provide accurate, context-aware, and verifiable answers by synthesizing information from a specialized knowledge base of research papers and technical documents.

---

## ✨ Features

- **Conversational Interface**  
  Ask questions in natural language. The system maintains context across interactions.

- **Retrieval-Augmented Generation (RAG)**  
  Goes beyond simple text generation by retrieving relevant documents before answering.

- **Multi-Source Verification**  
  Combines a **vector store** (FAISS) for semantic search with a **graph database** (Neo4j) for structured queries.

- **Transparent Reasoning**  
  Each answer includes:
  - Cited sources
  - Reasoning trace
  - Interactive graph of the query process

- **Real-time Streaming**  
  See answers and internal reasoning as they are generated.

- **Modular & Integratable**  
  Decoupled architecture enables integration into custom UIs, APIs, or CLI tools.

---

## ⚙️ System Architecture

| Component            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Streamlit UI**     | Web-based interface for interacting with the assistant                      |
| **CNTRagSystem**     | Core orchestrator handling the full query-processing pipeline                |
| **LLMInterface**     | Handles interaction with LLMs (e.g., OpenAI, Google Gemini)                  |
| **Vector Store**     | Uses FAISS to store document embeddings and perform semantic retrieval       |
| **Graph DB (Neo4j)** | Stores knowledge graphs representing entities and their relationships        |

---

## 🚀 Getting Started

### 1. Prerequisites

- Python **3.8+**
- Access to a **Neo4j** database instance
- API keys for a supported **LLM provider** (OpenAI, Google Gemini, etc.)

---

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### 3. Configuration

Create a `.env` file in the root directory and add the following:

```env
# LLM API Keys
OPENAI_API_KEY="your_openai_api_key"
GOOGLE_API_KEY="your_google_api_key"

# Neo4j Database Credentials
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_neo4j_password"

### 4. Knowledge Base Preaparation
To preapre the kowledge base (both knowledge graph and vector databse), just run 'python main.py'. This will create the full dataset.


For a complete walkthrough, code examples, and architectural details, please refer to the [Technical Integration Guide: CNT RAG System](CNT-Tech-Guide.pdf).