# Document-based Question Answering Application
This project implements a question-answering system for academic policies at Fulbright University Vietnam. It uses a combination of document retrieval and large language models to provide answers based on academic policy documents. The system is built with Streamlit for the user interface and leverages LangChain, HuggingFace, and Chroma for document processing and retrieval.


## Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (for LLM inference)
- At least 16GB of RAM and sufficient disk space for models (~15GB for Qwen2.5-7B)

TODO:
- [x] Implement full pipline of RAG: embedding, vectorstore and LLM.
- [x] Work on universal ways to preprocess data. For now it is done for AA policy.
- [x] Finetune different embedding model and LLM models.
- [x] Pick optimal chunk size for context.
- [x] Structure context generation.
- [x] Inspect why the current output context contains repetitive paragraphs.
- [x] Do research on possible (optimal) ways to index database.
- [x] Find ways to resolve conflicts among document (if any).
- [x] Employ Streamlit UI.
- [ ] Write report


### Resources

This model requires a large VRAM for single GPU. Currently, I am using Vast.ai with RTX A6000, 48 VRAM. Suggested VRAM on a single GPU is about 30GBs
