# Document-based Question Answering Application

### TODO:
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
