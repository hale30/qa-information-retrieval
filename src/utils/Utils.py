import datetime
import os
os.environ['HF_HOME'] = '/mnt/data/thomas/.cache' #Used to change where to save model. Uncomment this if you want to use default location
import fitz
import re
import gc
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings

def free_cuda_memory():
    """Function to release all CUDA memory and clear PyTorch cache."""
    # Collect garbage
    gc.collect()

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Reset memory allocations
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

def clean_capstone(pdf_path):
    """
    Cleans the Capstone PDF by removing unwanted lines and extracting sections.
    """
    doc = fitz.open(pdf_path)
    exact_removal = {
        "Fulbright University Vietnam Ground Floor, 105 Ton Dat Tien, Tan Phu, Quan 7, Ho Chi Minh City"
    }

    sections = []
    current_section_title = None
    current_section_content = []

    banned_verbs = [
        "Demonstrate", "Identify", "Effectively", "Select", "Conceptualize", "Communicate", "Produce",
        "Historical Reasoning", "Originality", "Communication", "Write a clear and answerable",
        "Critically analyze literature related to the topic", "Apply scientific methods",
        "Gain experience proposing", "Develop nuanced writing", "Explore and evaluate the pertinence",
        "Apply skills in the Literature major", "The language used is appropriate for", "Table of Contents",
        "List of Tables", "Fulbright Seminar (4 credits, optional) and Experiential Learning",
        "Analyze historical, cultural, social and contemporary issues in Vietnam",
        "Develop original and innovative projects",
        "Area Studies"
    ]

    # Match "1." on a line by itself
    numbered_only_pattern = re.compile(r"^\d+\.$")

    # Match section headers like "1. Introduction", "I. General Info" (excluding banned content)
    full_section_pattern = re.compile(
        r"^(\d+|[IVXLCDM]+)\.\s*(?!(" + "|".join(re.escape(verb) for verb in banned_verbs) + r")\b)[A-Z]"
    )

    for page in doc:
        raw_paragraphs = [p.strip() for p in page.get_text().split("\n") if p.strip()]
        filtered = []

        for p in raw_paragraphs:
            if p in exact_removal or p.lower().startswith("internal") or p.isdigit():
                continue
            if re.match(r"^\d+\s*\|\s*Page$", p) or re.match(r"^Page\s+\d+\s+of\s+\d+", p, re.IGNORECASE):
                continue
            filtered.append(p)

        i = 0
        while i < len(filtered):
            line = filtered[i]
            next_line = filtered[i + 1] if i + 1 < len(filtered) else ""

            # Case 1: "1." or "I." on one line, followed by capitalized title
            if numbered_only_pattern.match(line) and next_line and re.match(r"^[A-Z]", next_line):
                line = f"{line} {next_line}"
                i += 1  # Merge with the next line

            # Case 2: Full header on one line, and not a banned content line
            if full_section_pattern.match(line):
                if current_section_title:
                    joined_content = " ".join(current_section_content)
                    sections.append(f"{current_section_title}: {joined_content}")
                current_section_title = line
                current_section_content = []
            else:
                current_section_content.append(line)

            i += 1

    if current_section_title and current_section_content:
        joined_content = " ".join(current_section_content)
        sections.append(f"{current_section_title}: {joined_content}")

    return sections


def clean_majordescription(pdf_path):
    """
    Cleans the Major Description PDF by removing unwanted lines and extracting sections.
    """
    doc = fitz.open(pdf_path)
    exact_removal = {
        "Fulbright University Vietnam Ground Floor, 105 Ton Dat Tien, Tan Phu, Quan 7, Ho Chi Minh City"
    }

    sections = []
    current_section_title = None
    current_section_content = []
    wanted_section_titles = [
        "APPLIED MATHEMATICS",
        "ARTS AND MEDIA STUDIES",
        "COMPUTER SCIENCE",
        "ECONOMICS",
        "HUMAN-CENTERED ENGINEERING",
        "HISTORY",
        "PSYCHOLOGY",
        "INTEGRATED SCIENCES",
        "LITERATURE",
        "SOCIAL STUDIES",
        "VIETNAM STUDIES",
    ]

    for page in doc:
        raw_paragraphs = [p.strip() for p in page.get_text().split("\n") if p.strip()]
        filtered = []

        for p in raw_paragraphs:
            if p in exact_removal or p.isdigit() or p.lower().startswith("internal"):
                continue
            if re.match(r"^\d+\s*\|\s*Page$", p) or re.match(r"^Page\s+\d+\s+of\s+\d+", p, re.IGNORECASE):
                continue
            filtered.append(p)

        for line in filtered:
            if line in wanted_section_titles:  # Section header
                if current_section_title:  # Save the previous section
                    joined_content = " ".join(current_section_content)
                    sections.append(f"{current_section_title}: {joined_content}")
                current_section_title = line
                current_section_content = []
            else:
                current_section_content.append(line)

    # Save the last section
    if current_section_title and current_section_content:
        joined_content = " ".join(current_section_content)
        sections.append(f"{current_section_title}: {joined_content}")

    return sections


def clean_aapolicy(pdf_path):
    """
    Cleans the Academic Policy PDF by removing unwanted lines and extracting sections.
    """
    doc = fitz.open(pdf_path)
    exact_removal = {
        "Fulbright University Vietnam Ground Floor, 105 Ton Dat Tien, Tan Phu, Quan 7, Ho Chi Minh City"
    }

    sections = []
    current_section_title = None
    current_section_content = []

    for page in doc:
        raw_paragraphs = [p.strip() for p in page.get_text().split("\n") if p.strip()]
        filtered = []

        for p in raw_paragraphs:
            if p in exact_removal or p.isdigit() or p.lower().startswith("internal"):
                continue
            if re.match(r"^\d+\s*\|\s*Page$", p) or re.match(r"^Page\s+\d+\s+of\s+\d+", p, re.IGNORECASE):
                continue
            filtered.append(p)

        for line in filtered:
            if line.isupper():  # Section header
                if current_section_title:  # Save the previous section
                    joined_content = " ".join(current_section_content)
                    sections.append(f"{current_section_title}: {joined_content}")
                current_section_title = line
                current_section_content = []
            else:
                current_section_content.append(line)

    # Save the last section
    if current_section_title and current_section_content:
        joined_content = " ".join(current_section_content)
        sections.append(f"{current_section_title}: {joined_content}")

    return sections


def load_data(folder_path):
    filenames = os.listdir(folder_path)
    data = []
    for file in filenames:
        if "major" in file.lower():
            print("Major")
            file_path = os.path.join(folder_path, file)
            data.extend(clean_majordescription(file_path))
        elif "capstone" in file.lower():
            print("Capstone")
            file_path = os.path.join(folder_path, file)
            data.extend(clean_capstone(file_path))
        elif "academic" in file.lower():
            print("AA Policy")
            file_path = os.path.join(folder_path, file)
            data.extend(clean_aapolicy(file_path))
        else:
            print(f"File {file} not in categories. Skip!!!!")
            continue

    return data


def chunk_paragraphs(paragraphs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=512)
    return splitter.split_text("\n\n".join(paragraphs))

# === Step 3: Build vector store ===
def build_vectorstore(chunks, persist_path="./answer_all_policy/database/chroma_fulbright2", model_name = "Alibaba-NLP/gte-multilingual-base"):
    documents = [Document(page_content=chunk) for chunk in chunks]
    # print(type(documents[0]))
    embedding_model = HuggingFaceEmbeddings(model_name= model_name)
    # model_name_or_path = "Alibaba-NLP/gte-multilingual-base"
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_path
    )
    vectorstore.persist()
    return vectorstore

# === Step 4: Load local LLM ===
def load_local_llm(model_id="Qwen/Qwen2.5-7B-Instruct-1M"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, timeout=60)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

# === Step 5: Ask questions ===
def ask_question(llm_pipe, vectorstore, query, top_k=3):
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

    # print("\n=== PROMPT ===\n", prompt)

    response = llm_pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)[0]["generated_text"]
    # print("\n=== RESPONSE ===\n", response[len(prompt):].strip())
    return context, response[len(prompt):].strip()

# def concatenate_paragraphs(lines):
#     paragraphs = []
#     current_paragraph = ""
#
#     for line in lines:
#         stripped = line.strip()
#
#         # If it's a section header, start a new paragraph
#         if stripped and (stripped[0].isdigit() or stripped.startswith("•")):
#             if current_paragraph:
#                 paragraphs.append(current_paragraph)
#                 current_paragraph = stripped
#             # paragraphs.append(stripped)
#         # Continue building the paragraph
#         elif current_paragraph and (current_paragraph[-1] in ".:" or stripped[0].isupper()):
#             paragraphs.append(current_paragraph)
#             current_paragraph = stripped
#         else:
#             current_paragraph += " " + stripped if current_paragraph else stripped
#
#     if current_paragraph:
#         paragraphs.append(current_paragraph)
#
#     return paragraphs


def prepare_question():
    list_questions = [
        "Is it possible to graduate with double majors in HCE and Applied Mathematics? If yes, please provide a sample study plan.",
    ]
    return list_questions


def main():
    save_folder = "answer_just_major_old_new_policy"
    answer_path = os.path.join(save_folder, "answer")
    database_path = os.path.join(save_folder, "database")
    data_path = "/home/thomas/Downloads/qa-information-retrieval/data"
    os.makedirs(answer_path, exist_ok=True)
    os.makedirs(database_path, exist_ok=True)

    # Define embedding models to be tested
    embedding_models = [
        # "Alibaba-NLP/gte-multilingual-base",
        # "ibm-granite/granite-embedding-125m-english",
        # "NovaSearch/stella_en_400M_v5",
        # "jinaai/jina-embeddings-v3",
        "NovaSearch/jasper_en_vision_language_v1",
        "w601sxs/b1ade-embed"
    ]

    # Define LLM models to be tested
    llm_models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "allenai/Llama-3.1-Tulu-3-8B",
        "Qwen/Qwen2.5-7B-Instruct-1M",
        # "upstage/solar-pro-preview-instruct",
        # "Cran-May/tempmotacilla-cinerea-0308"
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

            print(f"-"*30 + f"Answer with {embedding_name} and {llm_name}" + "-"*30)

            # Load the selected LLM model
            try:
                llm_pipe = load_local_llm(llm_model)
            except torch.cuda.OutOfMemoryError:
                print(f"Cuda out of memory when loading model LLM {llm_name}!!!! Continue")

            # Prepare questions to ask the model
            list_questions = prepare_question()
            # Initialize markdown content for results
            markdown_content = f"""
# Experiment Results

## Model: {llm_model} with {embedding_model}
            """
            for question in list_questions:
                try:
                    # Get response from the LLM
                    context, response = ask_question(llm_pipe, vectorstore, question)
                    print(response)
                    print("-" * 50)

                    # Append results to markdown content
                    markdown_content += f"""
                
    ### Question
    {question}
    
    ### Context
    {context}
    
    ### Response
    {response}
    """
                except torch.cuda.OutOfMemoryError:
                    print("Cuda is out of memory!!! Continue to the next question.")
                    print("-"*50)
                    markdown_content += f"""
    ### Question
    {question}
    Cuda is out of memory!!!! Continue
    """

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


