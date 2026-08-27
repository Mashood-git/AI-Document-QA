from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")


# 1. Load the PDF
loader = PyPDFLoader("documents/sample_resume.pdf")
documents = loader.load()

print("Number of pages:", len(documents))


# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 3. Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

print("Creating embeddings...")


# 4. Store documents in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="document_qa"
)

print("Documents stored in ChromaDB successfully! ✅")


# 5. Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# 6. Ask a question
question = "Where did Mashood complete his Bachelor of Engineering?"


# 7. Retrieve relevant documents
# 7. Retrieve relevant documents
results = vector_store.similarity_search(
    question,
    k=5
)

print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================")

for i, document in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)



# 8. Combine retrieved chunks
context = "\n\n".join(
    document.page_content
    for document in results
)

# 9. Create RAG prompt
prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the information
provided in the document context.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- If the answer is not present in the context, say:
  "The information is not available in the document."
- Keep the answer concise.

Document Context:
{context}

User Question:
{question}

Answer:
"""

# 10. Ask Gemini
response = llm.invoke(prompt)

# 11. Display answer
print("\n==============================")
print("AI ANSWER")
print("==============================")
if isinstance(response.content, list):
    answer = response.content[0]["text"]
else:
    answer = response.content

print(answer)