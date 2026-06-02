import os

print(os.getcwd())
print(os.path.exists("documents/company_policy.pdf"))

from src.loader import load_pdf
from src.chunking import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.chatbot import ask_llm

print("Loading PDF...")

documents = load_pdf(
    "documents/company_policy.pdf"
)

print("Chunking Documents...")

chunks = split_documents(
    documents
)

print("Creating Embeddings...")

embeddings = get_embeddings()

print("Creating Vector Store...")

vectordb = create_vector_store(
    chunks,
    embeddings
)

print("Creating Retriever...")

retriever = get_retriever(
    vectordb
)

print("\nRAG System Ready!")
print("Type 'exit' to quit.")

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

    answer = ask_llm(prompt)

    print("\nAnswer:")
    print(answer)
