from utils import *
import torch
import os
from datetime import datetime

def prepare_paths(base_path):
    answer_path = os.path.join(base_path, "answers")
    database_path = os.path.join(base_path, "standard_database")
    data_path = os.path.join(base_path, "documents")

    os.makedirs(answer_path, exist_ok=True)
    os.makedirs(database_path, exist_ok=True)

    return answer_path, database_path, data_path

def process_embedding_model(embedding_model, chunks, database_path):
    embedding_name = embedding_model.split("/")[-1]
    print(f"[{datetime.now()}] Building vectorstore for: {embedding_name}")

    vectorstore = build_vectorstore(
        chunks=chunks,
        persist_path=os.path.join(database_path, embedding_name),
        model_name=embedding_model
    )
    print("Vectorstore built and persisted.")
    return embedding_name, vectorstore

def load_llm_safely(llm_model):
    llm_name = llm_model.split("/")[-1]
    try:
        print(f"[{datetime.now()}] Loading LLM: {llm_name}")
        llm_pipe = load_local_llm(llm_model)
        return llm_name, llm_pipe
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM when loading LLM {llm_name}. Skipping...")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading LLM {llm_name}: {e}. Skipping...")
    return llm_name, None

def answer_questions(llm_pipe, vectorstore, questions, llm_name, embedding_name):
    markdown_content = f"# Experiment Results\n## Model: {llm_name} with {embedding_name}\n"

    for i, question in enumerate(questions, start=1):
        try:
            print(f"[{datetime.now()}] Asking Q{i}: {question}")
            with torch.no_grad():
                context, response = ask_question(llm_pipe, vectorstore, question, top_k=5)
            print(response)
            print("-" * 50)
            markdown_content += f"""### Question {i}: {question}  
### Answer: 

{response}  

<details>  
<summary>References</summary>  

{context}  

</details> 

"""
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM during Q{i}. Skipping...")
            torch.cuda.empty_cache()
            markdown_content += f"""### Question {i}: {question}\n Cuda is out of memory!!!! Skipping\n"""
        except Exception as e:
            print(f"Error during Q{i}: {e}")
            markdown_content += f"""### Question {i}: {question}\n Error: {str(e)}\n"""

    return markdown_content

def save_results(markdown_content, answer_path, llm_name, embedding_name):
    file_path = os.path.join(answer_path, f"answers_with_{llm_name}_{embedding_name}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Results saved at: {file_path}")

def main():
    base_path = "/home/thomas/Downloads/qa-information-retrieval_2/data/"
    answer_path, database_path, data_path = prepare_paths(base_path)
    
    # Define embedding models to be tested
    embedding_models = [
        # "Alibaba-NLP/gte-multilingual-base",
        # "ibm-granite/granite-embedding-125m-english",
        # "NovaSearch/stella_en_400M_v5",
        "jinaai/jina-embeddings-v3",
        # "NovaSearch/jasper_en_vision_language_v1",
        # "w601sxs/b1ade-embed"
    ]
    
    # Define LLM models to be tested
    llm_models = [
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "allenai/Llama-3.1-Tulu-3-8B",
        "Qwen/Qwen2.5-7B-Instruct-1M",
    ]
    print(f"[{datetime.now()}] Loading and chunking data...")
    data = load_data(data_path)
    chunks = chunk_paragraphs(data)

    questions = prepare_question(os.path.join(base_path, "questions.json"))

    for embedding_model in embedding_models:
        embedding_name, vectorstore = process_embedding_model(embedding_model, chunks, database_path)

        for llm_model in llm_models:
            print("-" * 30 + f" {embedding_name} + {llm_model.split('/')[-1]} " + "-" * 30)

            llm_name, llm_pipe = load_llm_safely(llm_model)
            if llm_pipe is None:
                continue

            markdown_content = answer_questions(llm_pipe, vectorstore, questions, llm_name, embedding_name)

            del llm_pipe
            torch.cuda.empty_cache()
            free_cuda_memory()

            save_results(markdown_content, answer_path, llm_name, embedding_name)

if __name__ == "__main__":
    main()
