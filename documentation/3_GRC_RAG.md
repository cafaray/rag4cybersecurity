# GRC Data Protection Policy Retrieval System

## Overview
This project builds a Retrieval-Augmented Generation (RAG) pipeline for Governance, Risk, and Compliance (GRC) use cases, focusing on regulatory compliance, risk assessment, and data governance. The system allows querying global data protection laws and retrieving relevant legal frameworks efficiently.

## Features
- **Data ingestion**: Loads country-specific data protection policies.
- **Vector embedding**: Uses `sentence-transformers` to generate embeddings for regulatory text.
- **Vector database storage**: Stores embeddings in ChromaDB for fast similarity search.
- **Natural language querying**: Uses LlamaCpp as a local LLM for interactive Q&A.
- **Regulatory gap analysis**: Helps organizations compare policies across different jurisdictions.

## Setup Instructions

### 1. Load Regulatory Data
The system loads data protection policies from a CSV file containing country-specific regulations.

```python
import pandas as pd

# Load the Country Data Protection Policies dataset
data_protection = pd.read_csv('/data/country_plus_policies.csv')

print("Sample Data:\n", data_protection.head())
```

### 2. Preprocessing and Value Counts
To understand the dataset distribution, we check the value counts for different regulations.

```python
data_protection.Subject.value_counts()
```

### 3. Embedding Model Setup
Embeddings are crucial for searching regulatory text efficiently.

```python
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load a sentence transformer model
original_model = SentenceTransformer('all-MiniLM-L12-v2')
original_model.save('./')

# Reload model using LangChain wrapper
embedding_model_path = './'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

# Example embeddings
sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = original_model.encode(sentences)
```

### 4. Store Embeddings in ChromaDB
ChromaDB is used to store and retrieve embeddings for quick lookups.

```python
import chromadb

chromadb_path = './'
chroma_client = chromadb.Client()

vectordb = Chroma.from_documents(
    documents=data_protection,
    embedding=embedding_model,
    persist_directory=chromadb_path
)

# Persist vector database
vectordb.persist()
```

### 5. Querying the Regulatory Database
Using vector similarity search, we retrieve relevant policies.

```python
query_text = "Iceland data privacy"
results = vectordb.similarity_search_with_score(query_text)
```

### 6. Integrate Large Language Model (LLM) for Natural Language Queries
We integrate an LLM to process queries and return structured regulatory insights.

```python
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configure LLM
llm = LlamaCpp(
    model_path="./wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.gguf.bin",
    n_gpu_layers=35,
    n_ctx=2048,
    f16_kv=True,
    verbose=False
)

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=True,
    verbose=False
)
```

### 7. Example Queries

#### Iceland's Data Privacy Policy
```python
res = qa("What is Iceland's Data Privacy Policy?")
print(res)
```
##### Output:
```
Iceland's Data Privacy Policy outlines the legal framework for processing personal data.
```

#### GDPR Compliance Responsibilities
```python
res = qa("Who is liable to abide by GDPR?")
print(res)
```
##### Output:
```
Both controllers and processors must comply with GDPR, even outside the EU if processing EU citizen data.
```

#### Logging Standards Policy
```python
res = qa("What makes a good logging standard policy?")
print(res)
```
##### Output:
```
A good logging policy defines objectives, identifies loggable data, ensures compliance, and includes incident response guidelines.
```

## GRC Use Case Summary
This project helps organizations:
1. **Understand jurisdictional regulations**: Quickly retrieve relevant data protection laws for different countries.
2. **Compare compliance frameworks**: Identify differences between GDPR, NIST, and country-specific regulations.
3. **Automate regulatory reporting**: Generate structured reports for legal teams.
4. **Enhance risk assessment**: Map policies to security controls for governance.

## Future Enhancements
- Expand the dataset to include more industry-specific regulations.
- Implement a dashboard for interactive compliance insights.
- Improve LLM fine-tuning for legal-specific queries.

This project is an essential tool for **compliance officers, legal teams, and risk management professionals** seeking efficient ways to navigate regulatory complexities.

