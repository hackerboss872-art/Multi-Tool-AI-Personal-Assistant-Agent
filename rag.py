import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# -----------------------------
# Local LLM: Ollama + Qwen
# -----------------------------
def create_llm():
    return ChatOllama(
        model="qwen2.5:3b",
        temperature=0.2
    )


# -----------------------------
# Process PDF
# -----------------------------
def process_pdf(file_path="./resume.pdf"):
    if not os.path.exists(file_path):
        print(f"PDF not found: {file_path}")
        return None

    print("Loading PDF...")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        print("No text found in PDF.")
        return None

    print(f"PDF pages loaded: {len(docs)}")

    # Split PDF text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks were created.")
        return None

    # -----------------------------
    # Local HuggingFace Embeddings
    # -----------------------------
    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Chroma Vector Database
    # -----------------------------
    print("Creating Chroma vector database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="resume_rag"
    )

    return vectorstore


# -----------------------------
# Ask question from PDF
# -----------------------------
def ask_question(vectorstore, question):
    llm = create_llm()

    # Retrieve relevant PDF chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "I could not find relevant information in the PDF."

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the PDF context below.

If the answer is not present in the PDF, say:
"I could not find that information in the PDF."

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content


# -----------------------------
# Main program
# -----------------------------
def main():

    print("=" * 50)
    print("LOCAL PDF RAG AI")
    print("=" * 50)

    pdf_path = "./resume.pdf"

    vectorstore = process_pdf(pdf_path)

    if vectorstore is None:
        print("\nCould not create the PDF database.")
        return

    print("\nPDF RAG is ready!")
    print("Ask questions about your PDF.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer = ask_question(vectorstore, question)

            print("\nAI:", answer)
            print()

        except Exception as e:
            print("\nError:", e)
            print()


if __name__ == "__main__":
    main()