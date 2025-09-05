import pandas as pd
import chromadb
import re

from sentence_transformers import SentenceTransformer

# Deprecation warnings from LangChain + Chroma: move to langchain_huggingface
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader

# Load the MITRE ATT&CK datasets
mitre_data = pd.read_csv('./data/mitreembed_master_Chroma.csv', delimiter='\t')
cisa_data = pd.read_csv('./data/CISA_combo_features_new.csv')

print("MITRE Data Sample:\n", mitre_data.head())
print("CISA Data Sample:\n", cisa_data.head())
print(mitre_data.Source.value_counts())
# print(cisa_data.Source.value_counts())
print(mitre_data.keys())
# clean the body column:
mitre_data['Body'] = mitre_data['Body'].fillna("").astype(str)


# Initialize the Sentence Transformer
#   download embeddings model
original_model_path = './models/all-MiniLM-L12-v2'
original_model = SentenceTransformer('all-MiniLM-L12-v2')
# reload model using langchain wrapper
original_model.save(original_model_path)

embedding_model_path = './models/all-MiniLM-L12-v2'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
embeddings = model.encode(sentences)
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
# print(embeddings)

# Setup chromadb 
# define logic for embeddings storage
chromadb_path = './chromadb_data'

# for evaluating if running okay, uncomment below, it creates a chromadb client, 
# not required for langchain usage (Chroma.from_documents)
# chroma_client = chromadb.Client()
# print(chroma_client.get_version())

# assemble product documents in required format (id, text)
documents = (
  DataFrameLoader(
    mitre_data, 
    page_content_column='Body'
    )
    .load()  
  )

vectordb = Chroma.from_documents(
  documents=documents, 
  embedding=embedding_model, 
  persist_directory=chromadb_path, 
  #collection_name = 'CISA_MITRE'
  )
 
# Persist vector db to storage
# In Chroma ≥0.4.x, calling .persist() manually is deprecated. 
# Chroma now persists automatically when you set persist_directory.
# vectordb.persist()

#count documents 
# vectordb._collection.count()
print(f"Number of documents in the vector store: {vectordb._collection.count()}")

# Give a try and query the vector db
# query_text = "remote desktop attack"
query_text = "How to detect ransomware activity"
results = vectordb.similarity_search_with_score(query_text, k=3)
for doc, score in results:
    print(f"Score: {score:.4f}")
    # print(f"Content: {doc.metadata}\n")
    print(f"Body: {doc.page_content}\n")

# Scaling Considerations
# > When scaling your solution, consider the following factors:
# > - Storage Needs: The volume of data in ChromaDB.
# > - Performance: Using GPU-based instances for LLM operations.
# > - Cost Management: A typical GPU instance, such as AWS g4dn.4xlarge, can run around $500/month for 24x7 usage.

# Real-World Use Cases
# > By integrating ChromaDB with MITRE ATT&CK data, cybersecurity analysts can:
# > - Rapidly map alerts to known techniques.
# > - Cross-reference threat intelligence feeds.

# Conclusion:
# By leveraging ChromaDB and RAG frameworks, we can transform static cybersecurity data into dynamic, 
# actionable insights. This approach not only enhances threat detection and response but also empowers analysts 
# to make informed decisions quickly.