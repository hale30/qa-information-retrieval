# Document-based Question Answering Application

TODO:
- [x] Implement full pipline of RAG: embedding, vectorstore and LLM.
- [ ] Work on universal ways to preprocess data. For now it is done for AA policy.
- [ ] Finetune different embedding model and LLM models.
- [ ] Pick optimal chunk size for context.
- [ ] ...


## Note on running RAG_Pipeline.ipynb
This file uses the embedding models Alibaba-NLP/gte-multilingual-base, which requires the parameter trust_remote_control
when loading model set to True. Therefore, before running the notebook, first navigate to the SentenceTransformer.py file,
usually found at <path_to_virtual_env>/python3.10 (your python version)/site-packages/sentence_transformers/SentenceTransformer.py.
Go to the following part in the class SentenceTransformer, at around line 307, change:
```python
modules, self.module_kwargs = self._load_sbert_model(
                    model_name_or_path,
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                    config_kwargs=config_kwargs,
                )
```
Set trust_remote_code=True, then restart jupyter notebook and rerun again.

## Note on running RAG_Pipeline_big_model.ipynb

This model requires a large VRAM for single GPU. Currently, I am using Vast.ai with RTX A6000, 48 VRAM. Suggested VRAM 
on a single GPU is about 30GBs