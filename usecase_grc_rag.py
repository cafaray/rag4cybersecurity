import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader

from langchain_community.llms.gpt4all import GPT4All
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Load the Country Data Protection Policies dataset
# data_protection = pd.read_csv('./data/limited_country.csv')
data_protection = pd.read_csv('./data/data_protection_policies_rag.csv')
data_protection['Body'] = data_protection['Body'].fillna("").astype(str)
print("Sample Data:\n", data_protection.head())

# data_protection.Subject.value_counts()

# Load a sentence transformer model
embedding_model_path = './models/all-MiniLM-L12-v2/grc'
original_model = SentenceTransformer('all-MiniLM-L12-v2')
original_model.save(embedding_model_path)
# Reload model using LangChain wrapper
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

# Example embeddings
# sentences = ["This is an example sentence", "Each sentence is converted"]
# embeddings = original_model.encode(sentences)

chromadb_path = './chromadb_data'

documents = (
  DataFrameLoader(
    data_protection, 
    page_content_column='Body'
    )
    .load()  
  )

vectordb = Chroma.from_documents(
  documents=documents, 
  embedding=embedding_model, 
  persist_directory=chromadb_path, 
)

# Example of using the vector store
query_text = "Iceland data privacy"
results = vectordb.similarity_search_with_score(query_text, k=3)
for doc, score in results:
    print(f"Score: {score:.4f}")
    # print(f"Content: {doc.metadata}\n")
    print(f"Body: {doc.page_content}\n")

# INTEGRATING THE LLM AND RAG

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(
    # model="./models/ggml-gpt4all-j-v1.3-groovy.bin", 
    model="./data/gpt4all-falcon-newbpe-q4_0.gguf", 
    n_threads=8, 
    verbose=True,
    callback_manager=callback_manager
)

# Setup ChromaDB vector store
chromadb_path = './chromadb_data'
embedding_model_path = './models/all-MiniLM-L12-v2'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

vectordb = Chroma(
  persist_directory=chromadb_path, 
  embedding_function=embedding_model
)

#retrievers
retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    #retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    retriever=retriever,
    return_source_documents=True,
    verbose=False,
)

# Example query
query = "What are the key data protection regulations in Iceland?"
response = qa.invoke({  "query": query})

print("Answer:", response["result"])
print("\n\nSource Documents:")
for document in response["source_documents"]:
    print("\n")
    print(document.metadata)
    print(document.page_content[:500])  # Print the first 500 characters of the document content

res = qa("What is Iceland's Data Privacy Policy?")
print(res)

res = qa("Who is liable to abide by GDPR?")
print(res)

res = qa("What makes a good logging standard policy?")
print(res)