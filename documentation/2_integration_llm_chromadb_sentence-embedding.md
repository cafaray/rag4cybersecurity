# Integrating LLM with ChromaDB and Sentence Embedding Models

## Overview

This guide will walk you through integrating a Large Language Model (LLM) with ChromaDB and a sentence embedding model using the WizardLM-13B model from Hugging Face. We will leverage LangChain to build a retrieval-augmented generation (RAG) pipeline to efficiently retrieve relevant data and generate responses.

## Prerequisites

Ensure you have the following installed:

```bash
pip install langchain==0.0.173 chromadb==0.3.23 pypdf==3.8.1 pygpt4all==1.1.0 
```

Make sure you have the model file downloaded from [WizardLM-13B Hugging Face Repo](https://huggingface.co/TheBloke/WizardLM-13B-V1-1-SuperHOT-8K-GGML):

- `wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.gguf.bin`

---

## Step 1: Load Necessary Libraries

```python
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
```

---

## Step 2: Load and Configure the LLM

```python
# Configure LLM with GPU layers
n_gpu_layers = 35
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="../wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.gguf.bin",
    
    n_gpu_layers=n_gpu_layers,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True to avoid memory issues
    callback_manager=callback_manager,
    verbose=False,
)
```

---

## Step 3: Load Langchain with LLM for RetrievalQA

```python
#https://medium.com/p/9f890e6960f3
# Successfully uninstalled langchain-0.0.198

from langchain import PromptTemplate, LLMChain

chromadb_path = './'

embedding_model_path = './'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

vectordb = Chroma(persist_directory=chromadb_path, embedding_function=embedding_model)
#retrievers

retriever = vectordb.as_retriever()

from langchain.chains import RetrievalQA

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    verbose=False,
)
```

---

## Step 4: Engage and ask questions

```python
res = qa("Explain which MITRE technique aligns with this signature: \
keywords': ['attempt to execute code on stack by', 'FTP LOGIN FROM .* 0bin0sh', 'rpc.statd[\\d+]: \
gethostbyname error for', 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'], 'condition': 'keywords")
res


)
```

---

## Step 5: Querying the System

```python
query = "What is the MITRE ATT&CK technique for remote desktop attacks?"
response = qa.run(query)
print("Response:", response)
```

---

## Step 6: Scaling Considerations

When scaling your solution, consider the following factors:

- **Storage Needs:** Plan according to the volume of data in ChromaDB.
- **Performance:** Use GPU-based instances for LLM operations.
- **Cost Management:** Monitor infrastructure costs such as AWS GPU instances.

---

## Step 7: Real-World Use Cases

By integrating ChromaDB with LLMs, cybersecurity analysts can:

1. Quickly map alerts to known techniques.

---

## Alternative Integration: OpenAI LLM with API Key

If you’d prefer to leverage OpenAI’s models (e.g., GPT‑3.5‑turbo) instead of the local WizardLM, you can use the following code block. Make sure you have your OpenAI API key set in your environment (or replace the placeholder below):

```python
import os
from langchain_openai import ChatOpenAI  # Ensure you installed langchain-openai with: pip install -U langchain-openai
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Retrieve your OpenAI API key from the environment variable "OPENAI_API_KEY"
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    # Alternatively, you can set it directly here (not recommended for production)
    openai_api_key = "YOUR_OPEN_AI_KEY_HERE"

# Instantiate the ChatOpenAI LLM with your API key
llm_openai = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0,
)

# Reuse the same embedding model and ChromaDB configuration
chromadb_path = './'
embedding_model_path = './'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
vectordb = Chroma(persist_directory=chromadb_path, embedding_function=embedding_model)

# Create a retriever from the vector database
retriever = vectordb.as_retriever(search_kwargs={"k": 1})

# Build the RetrievalQA chain using the OpenAI model
qa_openai = RetrievalQA.from_chain_type(
    llm=llm_openai,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=True,
    verbose=False,
)


# Example query using the OpenAI-powered chain with the new method
query_openai = "What is the MITRE ATT&CK technique for remote desktop attacks?"
response_openai = qa_openai.invoke(query_openai)

# Print only the main result (you can also inspect source_documents if needed)
print("Response from OpenAI LLM:", response_openai["result"])
```

Reponse

```
Response from OpenAI LLM: The MITRE ATT&CK technique for remote desktop attacks is T1076 - Remote Desktop Protocol. This technique covers the use of Remote Desktop Protocol (RDP) to gain unauthorized access to a system or move laterally within an environment.
```

---

## Conclusion

Congratulations! You have successfully integrated an LLM with ChromaDB and a sentence embedding model. This setup allows you to harness RAG to improve cybersecurity operations by retrieving relevant data and generating insightful responses.

Stay tuned for further optimizations and advanced configurations!"}
