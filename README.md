# RAG for Cybersecurity

This repository contains the minimal files to formulate an integration between RAG, LLM and Cibersecurity. 

> Original (forked) repository: [rag-for-cybersecurity](https://github.com/cafaray/rag-for-cybersecurity)

To run: 

- app.py: Load the files located at `data` folder, transform into embedded format and generate the models, create the chromadb database and load the embedded data. Execute some test queries. 

- integrate_llm_local.py: Integrate a local LLM to the app.py script. There are two LLM proposals: [wizardlm-13b-v1.1](https://huggingface.co/TheBloke/WizardLM-13B-V1-1-SuperHOT-8K-GGML/blob/main/wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin) and [gpt4all-falcon-newbpe-q4_0](https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf); the first one is recommended if you are using a python 11 version, and the second one for a python 12 version. Of course, some changes should be done if you decide to use python-11.

- integrate_llm_openai.py: Integrate a remote LLM to the app.py script. It use the openai formula.

- usecase_grc_rag.py: A complete use case for security and compliance.  
