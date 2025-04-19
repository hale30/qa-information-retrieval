from utils import *
import torch
import os

def main():
    save_folder = "/home/thomas/Downloads/qa-information-retrieval_2/data/" # Change the name of the folder to save the output
    answer_path = os.path.join(save_folder, "answer_1")
    database_path = os.path.join(save_folder, "database_no_student_journey")
    data_path = "/home/thomas/Downloads/qa-information-retrieval_2/data/documents" # Change the path to your data file
    os.makedirs(answer_path, exist_ok=True)
    os.makedirs(database_path, exist_ok=True)

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

    # Load and preprocess the PDF document
    data = load_data(data_path)
    chunks = chunk_paragraphs(data)  # Chunk text into manageable pieces

    for embedding_model in embedding_models:
        embedding_name = embedding_model.split("/")[-1]

        # Build vectorstore for each embedding model
        vectorstore = build_vectorstore(chunks, persist_path=os.path.join(database_path, embedding_name),
                                        model_name=embedding_model)
        print("✅ Vectorstore built and persisted.")

        for llm_model in llm_models:
            llm_name = llm_model.split("/")[-1]

            print(f"-" * 30 + f"Answer with {embedding_name} and {llm_name}" + "-" * 30)

            # Load the selected LLM model
            try:
                llm_pipe = load_local_llm(llm_model)
            except torch.cuda.OutOfMemoryError:
                print(f"Cuda out of memory when loading model LLM {llm_name}!!!! Continue")

            # Prepare questions to ask the model
            list_questions = prepare_question("/home/thomas/Downloads/qa-information-retrieval_2/data/questions_wrong_context.json")
            # Initialize markdown content for results
            markdown_content = f"""# Experiment Results\n## Model: {llm_model} with {embedding_model}\n"""
            for i, question in enumerate(list_questions, start=1):
                try:
                    # Get response from the LLM
                    context, response = ask_question(llm_pipe, vectorstore, question, top_k = 5)
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
                    print("Cuda is out of memory!!! Continue to the next question.")
                    print("-" * 50)
                    markdown_content += f"""### Question {i}: {question}\n Cuda is out of memory!!!! Continue"""

            # Free up CUDA memory after processing each model
            free_cuda_memory()

            # Save markdown file with experiment results
            # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = os.path.join(answer_path, f"answers_with_{llm_name}_{embedding_name}.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"Markdown file saved at: {file_path}")


if __name__ == "__main__":
    main()