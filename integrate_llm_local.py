import chromadb
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.llms import LlamaCpp
from langchain_community.llms.gpt4all import GPT4All

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# This script demonstrates how to set up a Retrieval-Augmented Generation (RAG) system
# using LangChain, ChromaDB, and a local LLM (GPT4All or LlamaCpp).
# The example uses a dataset of country data protection policies.
# - Download and save the embedding model locally into ./data folder
#   - curl -L -O https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf
#   - curl -L -O https://huggingface.co/TheBloke/WizardLM-13B-V1-1-SuperHOT-8K-GGML/blob/main/wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin

# Configure LLM with GPU layers
# n_gpu_layers = 35
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Configure GPT4All (CPU-friendly, no llama-cpp needed)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(
    # model="./models/ggml-gpt4all-j-v1.3-groovy.bin", 
    model="./data/gpt4all-falcon-newbpe-q4_0.gguf", 
    n_threads=8, 
    verbose=True,
    callback_manager=callback_manager
)
# llm = LlamaCpp(
#    model_path="./data/wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin",    
#    n_gpu_layers=n_gpu_layers,
#    n_ctx=2048,
#    f16_kv=True,  # MUST set to True to avoid memory issues
#    callback_manager=callback_manager,
#    verbose=False,
# )

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

# Build prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Use three sentences maximum. Keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer. 

{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    #retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    retriever=retriever,
    return_source_documents=True,
    verbose=False,
)

# Engage and ask questions

res = qa("Explain which MITRE technique aligns with this signature: \
keywords': ['attempt to execute code on stack by', 'FTP LOGIN FROM .* 0bin0sh', 'rpc.statd[\\d+]: \
gethostbyname error for', 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'], \
'condition': 'keywords")


print("MITRE Signature Response:", res)

# Querying the System
query = "What is the MITRE ATT&CK technique for remote desktop attacks?"
# response = qa.run(query)
response = qa.invoke({  "query": query})

# print("Response:", response)
print("Answer:", response["result"])
print("\n\nSource Documents:")
for document in response["source_documents"]:
    print("\n")
    print(document.metadata)
    print(document.page_content[:500])  # Print the first 500 characters of the document content