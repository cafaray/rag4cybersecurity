# Uploading Data to ChromaDB

_Original version from [LinkedInLearning](https://raw.githubusercontent.com/LinkedInLearning/rag-for-cybersecurity-use-cases-and-implementation-4292591/refs/heads/02_02/Chapter2/Usecase1.md)_

## Objective

This guide walks through the process of uploading MITRE ATT&CK and CISA Advisories data into Chroma, a high-performance vector database, using Python. It aligns with the MITREembed project and supports Retrieval-Augmented Generation (RAG) in cybersecurity.

## Prerequisites

Ensure you have the following installed:

1. **Python 3.12+**  
2. **ChromaDB** (`pip install Chroma==0.2.0`)  
3. **Pandas** (`pip install pandas==2.0.2`)
4. **Transformers** (`pip install sentence-transformers==3.1.1`)
5. **MITRE ATT&CK Data:** `mitreembed_master_Chroma.csv` and `CISA_combo_features_new.csv`  
6. **CPU Machine** (Recommended for ChromaDB setup, GPU for LLM integration)

---

## Step 1: Set Up the Environment

Install the required libraries:

```bash
pip install chromadb pandas transformers
```

---

## Step 2: Load the MITRE ATT&CK Data

```python
import pandas as pd

# Load the MITRE ATT&CK datasets
mitre_data = pd.read_csv('mitreembed_master_Chroma.csv')
cisa_data = pd.read_csv('CISA_combo_features_new.csv')

print("MITRE Data Sample:\n", mitre_data.head())
print("CISA Data Sample:\n", cisa_data.head())
print(mitre_data.Source.value_counts())
```

---

## Step 3: Initialize the Sentence Transformer

```python
from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
# download embeddings model
original_model = SentenceTransformer('all-MiniLM-L12-v2')
# reload model using langchain wrapper
original_model.save('./')

embedding_model_path = './'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
embeddings = model.encode(sentences)
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
print(embeddings)
```

---

## Step 4: Setup ChromaDB

```python
#setup chromadb 

from langchain.vectorstores import Chroma
# define logic for embeddings storage
chromadb_path = './'
import chromadb
chroma_client = chromadb.Client()
print(chroma_client.get_version())

from sentence_transformers import SentenceTransformer
 
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import re
 
import pandas as pd
# assemble product documents in required format (id, text)
documents = (
  DataFrameLoader(
    mitre_data,
    page_content_column='Body'
    )
    .load()
  )
```

---

## Step 5: Define Logic for embeddings storage

```python
chromadb_path = './'

vectordb = Chroma.from_documents(
  documents=documents, 
  embedding=embedding_model, 
  persist_directory=chromadb_path, 
  #collection_name = 'CISA_MITRE'
  )
 
# persist vector db to storage
vectordb.persist()
```

## Step 6: Check the number of collections in the vectorDB

```python
#count documents 
vectordb._collection.count()
```

## Step 7: Query Chroma

```python
query_text = "remote desktop attack"
vectordb.similarity_search_with_score(query_text)
print("Top matching techniques:", results['metadatas'])
```

---

## Step 7: Inspect a record

```python
#examine a vector db record
rec= vectordb._collection.peek(1)
print('Metadatas:  ', rec['metadatas'])
print('Documents:  ', rec['documents'])
print('ids:        ', rec['ids']) 
print('embeddings: ', rec['embeddings'])
```

---

## Step 8: Scaling Considerations

When scaling your solution, consider the following factors:

- **Storage Needs:** The volume of data in ChromaDB.
- **Performance:** Using GPU-based instances for LLM operations.
- **Cost Management:** A typical GPU instance, such as AWS `g4dn.4xlarge`, can run around $500/month for 24x7 usage.

---

## Step 9: Real-World Use Cases

By integrating ChromaDB with MITRE ATT&CK data, cybersecurity analysts can:

1. Rapidly map alerts to known techniques.
2. Cross-reference threat intelligence feeds.

---

## Step 10: Next Steps

Now that you've successfully uploaded and queried MITRE ATT&CK data in Chroma, consider:

- **Experimenting with different embedding models** for improved accuracy.

---

## Conclusion

By leveraging ChromaDB and RAG frameworks, we can transform static cybersecurity data into dynamic, actionable insights. Stay tuned for more advanced sessions on scaling your solution and optimizing performance.
