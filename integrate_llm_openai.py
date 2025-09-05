import os
from langchain_openai import ChatOpenAI  # Ensure you installed langchain-openai with: pip install -U langchain-openai
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Retrieve your OpenAI API key from the environment variable "OPENAI_API_KEY"
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    # Alternatively, you can set it directly here (not recommended for production)
    print("Please set the OPENAI_API_KEY environment variable.")

# Instantiate the ChatOpenAI LLM with your API key
llm_openai = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0,
)

# Reuse the same embedding model and ChromaDB configuration
chromadb_path = './chromadb_data'
embedding_model_path = './models/all-MiniLM-L12-v2'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
vectordb = Chroma(persist_directory=chromadb_path, embedding_function=embedding_model)

# Create a retriever from the vector database
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

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