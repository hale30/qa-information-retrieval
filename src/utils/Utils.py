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

#
# def load_and_clean_pdf(pdf_path):
#     if not os.path.isdir(pdf_path):
#         filenames = [pdf_path]
#     else:
#         filenames = os.listdir(pdf_path)
#     all_paragraphs = []
#
#
#     for pdf_file in filenames:
#         doc = fitz.open(pdf_file)
#         exact_removal = {
#             "Fulbright University Vietnam Ground Floor, 105 Ton Dat Tien, Tan Phu, Quan 7, Ho Chi Minh City"
#         }
#
#         for i, page in enumerate(doc):
#             # print(repr(page.get_text()))
#             raw_paragraphs = [p.strip() for p in page.get_text().split("\n") if p.strip()]
#             filtered = []
#             # print(raw_paragraphs)
#             # print("-"*30)
#
#             for p in raw_paragraphs:
#                 if p in exact_removal:
#                     continue
#                 if p.isdigit():
#                     continue
#                 if p.lower().startswith("internal"):
#                     continue
#                 if re.match(r"^\d+\s*\|\s*Page$", p):
#                     continue
#                 if re.match(r"^Page\s+\d+\s+of\s+\d+", p, re.IGNORECASE):
#                     continue
#                 filtered.append(p)
#
#             if i > 0 and filtered:
#                 first_word = filtered[0].split()[0] if filtered[0].split() else ""
#                 if first_word and not first_word[0].isupper():
#                     all_paragraphs[-1] += " " + filtered[0]
#                     filtered = filtered[1:]
#
#             all_paragraphs.extend(filtered)
#
#     return all_paragraphs

def load_and_clean_pdf(pdf_path):
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

def chunk_paragraphs(paragraphs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    return splitter.split_text("\n\n".join(paragraphs))

# === Step 3: Build vector store ===
def build_vectorstore(chunks, persist_path="./chroma_fulbright2", model_name = "Alibaba-NLP/gte-multilingual-base"):
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
    os.makedirs("Answers", exist_ok=True)
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

    # print("\n=== PROMPT ===\n", prompt)

    response = llm_pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)[0]["generated_text"]
    # print("\n=== RESPONSE ===\n", response[len(prompt):].strip())
    return context, response[len(prompt):].strip()

def concatenate_paragraphs(lines):
    paragraphs = []
    current_paragraph = ""

    for line in lines:
        stripped = line.strip()

        # If it's a section header, start a new paragraph
        if stripped and (stripped[0].isdigit() or stripped.startswith("•")):
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = stripped
            # paragraphs.append(stripped)
        # Continue building the paragraph
        elif current_paragraph and (current_paragraph[-1] in ".:" or stripped[0].isupper()):
            paragraphs.append(current_paragraph)
            current_paragraph = stripped
        else:
            current_paragraph += " " + stripped if current_paragraph else stripped

    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs


def prepare_question():
    list_questions = []
    list_questions.append("Can you differentiate a cross-listed course and elective applied course in the case of a double major?")
    list_questions.append("I would like to ask for the capstone withdrawal policy. What will be the impact when I choose to drop the capstone before the Fall 2025 term starts? Will there be any penalties associated with withdrawing, such as a 'W' notation on my transcript?")
    list_questions.append("Questions related to course history, in which one course can have multiple IDs? Like Advanced Deep Learning course. How would it be handled among different cohorts?")
    # list_questions.append("I have taken Microeconomic Analysis and Macroeconomic Analysis in Fall 2023. Am I eligible to take Financial Economics in Spring 2024?")
    # list_questions.append("If I have waived the course Computer Science I, but I want to declare Computer Science major, do I have to take Computer Science I?")
    # list_questions.append("If I have taken the following courses: Mathematical Statistics, Abstract Algebra, Linear Algebra, Multivariable Calculus, Calculus, Differential Equation, Stochastic Calculus, Statistical Learning, Discrete Mathematics, Probability, am I eligible to declare a minor in Applied Mathematics? What about majoring in Applied Mathematics?")
    # list_questions.append("I have completed these 100-level courses: Introduction to Art History and Theory, Introduction to Film History and Theory. Am I eligible to start a Capstone in my major Arts and Media Studies?")
    # list_questions.append("Is it possible that I write a book as my History Capstone?")
    # list_questions.append("Given I have finished these subjects: Advanced Deep Learning, Machine Learning, Deep Learning, Database. Are they sufficient for the Advanced Courses requirement of the computer science major?")
    list_questions.append("I have a question if I take a gap year, will my financial aid be canceled? If yes, can I still re-apply for the financial aid program when I get back to school? Thank you.")
    list_questions.append("I am aware that Fulbright has to organize online classes instead of on-campus ones. Therefore, will there be any change in the tuition fee?")
    # list_questions.append("If I major in Economics, is it sufficient that I take Scholar Development to fulfill Experiential Learning requirements?")
    # list_questions.append("Is it sufficient for me to complete 4 credits in EL if I major in Engineering?")
    # list_questions.append("Can I take all four engineering foundation courses and only two intermediate engineering courses?")
    # list_questions.append("If I have completed the course Intro to AI, will CS207 - Object-Oriented Analysis and Design and CS211 – Operating Systems suffice to fill the rest two intermediate courses?")
    # list_questions.append("Does completing both Microeconomics and Macroeconomics satisfy the core theory requirement, or is an additional course needed?")
    # list_questions.append("If I want to major in Engineering, can I choose other programming courses besides Computer Science I and Computer Science II, for instance Introduction to data visualization?")
    list_questions.append("Given I took Discrete Mathematics in Spring 2022 and earned B, then the same course in Spring 2024 and earned C-, which grades will be kept?")
    list_questions.append("Can I register 300-level math courses as exploratory courses E4?")
    list_questions.append("Can I register for more than 20 credits in a semester?")
    list_questions.append("What is the maximum number of times I can retake a course to improve my grade, and how does it affect my transcript?")
    # list_questions.append("For a student doing the literature capstone, if a student includes charts or tables, do they count toward the word limit?")
    # list_questions.append("If I failed Capstone I in the Fall 2024, can I redo it in Spring 2025?")
    # list_questions.append("If I do Literature capstone in Decolonial studies, can it be approved?")
    # list_questions.append("I am a Co24 majoring in Computer Science and taking the Database course. Does it count as major courses (difference in flowchart of CS for Co24 and Co25 onwards)?")
    list_questions.append("Can Discrete Mathematics be counted for both Computer Science and Applied Mathematics majors?")
    return list_questions


def main():
    # Define embedding models to be tested
    embedding_models = [
        "Alibaba-NLP/gte-multilingual-base",
        "ibm-granite/granite-embedding-125m-english",
        "NovaSearch/stella_en_400M_v5",
        "jinaai/jina-embeddings-v3",

    ]

    # Define LLM models to be tested
    llm_models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "allenai/Llama-3.1-Tulu-3-8B",
        "upstage/solar-pro-preview-instruct",
        "Qwen/Qwen2.5-14B-Instruct-1M"
    ]

    # Load and preprocess the PDF document
    pdf_file = load_and_clean_pdf("/home/thomas/Downloads/qa-information-retrieval/data/Academic-Policy_V5.1.pdf")
    # test_paragraph = concatenate_paragraphs(pdf_file)  # Concatenate broken lines into paragraphs
    # chunks = chunk_paragraphs(test_paragraph)  # Chunk text into manageable pieces
    chunks = chunk_paragraphs(pdf_file)  # Chunk text into manageable pieces


    for embedding_model in embedding_models:
        embedding_name = embedding_model.split("/")[-1]

        # Build vectorstore for each embedding model
        vectorstore = build_vectorstore(chunks, persist_path="./chroma_fulbright_" + embedding_name,
                                        model_name=embedding_model)
        print("✅ Vectorstore built and persisted.")

        for llm_model in llm_models:
            llm_name = llm_model.split("/")[-1]

            # Load the selected LLM model
            llm_pipe = load_local_llm(llm_model)

            # Prepare questions to ask the model
            list_questions = prepare_question()
            # Initialize markdown content for results
            markdown_content = f"""
# Experiment Results

## Model: {llm_model} with {embedding_model}
            """

            for question in list_questions:
                # Get response from the LLM
                context, response = ask_question(llm_pipe, vectorstore, question)
                print(response)

                # Append results to markdown content
                markdown_content += f"""
            
### Question
{question}

### Context
{context}

### Response
{response}
"""

            # Free up CUDA memory after processing each model
            free_cuda_memory()

            # Save markdown file with experiment results
            # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = os.path.join("Answers", f"answers_with_{llm_name}_{embedding_name}.md")
            os.makedirs("Answers", exist_ok=True)  # Ensure directory exists
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"Markdown file saved at: {file_path}")


if __name__ == "__main__":
    main()


