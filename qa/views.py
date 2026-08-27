import os
import uuid

from django.shortcuts import render

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma


load_dotenv()


def home(request):

    message = ""
    answer = ""
    sources = []

    # ==================================================
    # GET CURRENT DOCUMENT INFORMATION FROM SESSION
    # ==================================================

    document_id = request.session.get("document_id")
    original_filename = request.session.get("original_filename")

    # ==================================================
    # HANDLE POST REQUEST
    # ==================================================

    if request.method == "POST":

        # ==================================================
        # 1. HANDLE PDF UPLOAD
        # ==================================================

        uploaded_file = request.FILES.get("document")

        if uploaded_file:

            # ==================================================
            # VALIDATE FILE TYPE
            # ==================================================

            if not uploaded_file.name.lower().endswith(".pdf"):

                message = "Please upload a PDF file only."

            else:

                # ==================================================
                # GENERATE UNIQUE DOCUMENT ID
                # ==================================================

                document_id = str(uuid.uuid4())

                # Store document information in session
                request.session["document_id"] = document_id

                original_filename = uploaded_file.name

                request.session["original_filename"] = (
                    original_filename
                )

                # ==================================================
                # CREATE UPLOAD DIRECTORY
                # ==================================================

                upload_dir = "uploads"

                os.makedirs(
                    upload_dir,
                    exist_ok=True
                )

                # ==================================================
                # CREATE UNIQUE STORED FILENAME
                # ==================================================

                stored_filename = (
                    f"{document_id}.pdf"
                )

                file_path = os.path.join(
                    upload_dir,
                    stored_filename
                )

                # ==================================================
                # SAVE PDF
                # ==================================================

                with open(
                    file_path,
                    "wb+"
                ) as destination:

                    for chunk in uploaded_file.chunks():

                        destination.write(chunk)

                print("\n==============================")
                print("PDF UPLOADED")
                print("==============================")

                print(
                    "Original filename:",
                    original_filename
                )

                print(
                    "Stored filename:",
                    stored_filename
                )

                print(
                    "Document ID:",
                    document_id
                )

                # ==================================================
                # LOAD PDF
                # ==================================================

                loader = PyPDFLoader(
                    file_path
                )

                documents = loader.load()

                print(
                    "Number of pages:",
                    len(documents)
                )

                # ==================================================
                # ADD DOCUMENT METADATA
                # ==================================================

                for document in documents:

                    document.metadata[
                        "document_id"
                    ] = document_id

                    document.metadata[
                        "original_filename"
                    ] = original_filename

                # ==================================================
                # SPLIT DOCUMENT INTO CHUNKS
                # ==================================================

                text_splitter = RecursiveCharacterTextSplitter(

                    chunk_size=500,

                    chunk_overlap=100

                )

                chunks = (
                    text_splitter.split_documents(
                        documents
                    )
                )

                print(
                    "Number of chunks:",
                    len(chunks)
                )

                # ==================================================
                # CREATE EMBEDDINGS
                # ==================================================

                embeddings = (
                    GoogleGenerativeAIEmbeddings(
                        model="models/gemini-embedding-001"
                    )
                )

                print(
                    "Creating embeddings..."
                )

                # ==================================================
                # CREATE DOCUMENT-SPECIFIC CHROMA
                # ==================================================

                collection_name = (
                    f"document_{document_id}"
                )

                Chroma.from_documents(

                    documents=chunks,

                    embedding=embeddings,

                    collection_name=collection_name,

                    persist_directory="chroma_db"

                )

                print(
                    "Documents stored in ChromaDB successfully!"
                )

                message = (
                    "Document uploaded and "
                    "processed successfully! ✅"
                )

        # ==================================================
        # 2. HANDLE QUESTION
        # ==================================================

        question = request.POST.get(
            "question"
        )

        if question:

            print("\n==============================")
            print("QUESTION")
            print("==============================")

            print(question)

            # ==================================================
            # CHECK IF DOCUMENT EXISTS
            # ==================================================

            if not document_id:

                answer = (
                    "Please upload a document "
                    "before asking a question."
                )

            else:

                # ==================================================
                # CREATE EMBEDDINGS
                # ==================================================

                embeddings = (
                    GoogleGenerativeAIEmbeddings(
                        model="models/gemini-embedding-001"
                    )
                )

                # ==================================================
                # CONNECT TO DOCUMENT-SPECIFIC CHROMA
                # ==================================================

                collection_name = (
                    f"document_{document_id}"
                )

                vector_store = Chroma(

                    collection_name=collection_name,

                    persist_directory="chroma_db",

                    embedding_function=embeddings

                )

                # ==================================================
                # RETRIEVE RELEVANT CHUNKS WITH SCORES
                # ==================================================

                results_with_scores = (
                    vector_store.similarity_search_with_score(

                        question,

                        k=5

                    )
                )

                # ==================================================
                # APPLY RETRIEVAL SCORE THRESHOLD
                # ==================================================

                SCORE_THRESHOLD = 0.75

                filtered_results = []

                print(
                    "\n=============================="
                )

                print(
                    "RETRIEVED CHUNKS"
                )

                print(
                    "=============================="
                )

                for document, score in results_with_scores:

                    print(
                        "\n------------------------------"
                    )

                    print(
                        "Similarity Score:",
                        score
                    )

                    print(
                        "Content:"
                    )

                    print(
                        document.page_content
                    )

                    # ==================================================
                    # LOWER DISTANCE = BETTER MATCH
                    # ==================================================

                    if score <= SCORE_THRESHOLD:

                        filtered_results.append(
                            (
                                document,
                                score
                            )
                        )

                # ==================================================
                # GET DOCUMENTS AFTER FILTERING
                # ==================================================

                results = [

                    document

                    for document, score
                    in filtered_results

                ]

                print(
                    "\n=============================="
                )

                print(
                    "FILTERED RESULTS:",
                    len(results)
                )

                print(
                    "=============================="
                )

                # ==================================================
                # CHECK RETRIEVAL RESULTS
                # ==================================================

                if not results:

                    answer = (
                        "The information is not available "
                        "in the document."
                    )

                    print(
                        "\nNo relevant chunks found."
                    )

                else:

                    # ==================================================
                    # COMBINE RETRIEVED CONTEXT
                    # ==================================================

                    context_parts = []

                    for document in results:

                        context_parts.append(
                            document.page_content
                        )

                    context = "\n\n".join(
                        context_parts
                    )

                    # ==================================================
                    # GET SOURCE INFORMATION
                    # ==================================================

                    sources = []

                    for document in results:

                        source = (
                            document.metadata.get(

                                "original_filename",

                                original_filename
                                or "Unknown document"

                            )
                        )

                        page = (
                            document.metadata.get(
                                "page",
                                0
                            )
                        )

                        # PyPDFLoader page numbers
                        # start from 0

                        page_number = (
                            int(page) + 1
                        )

                        source_info = (
                            source,
                            page_number
                        )

                        if source_info not in sources:

                            sources.append(
                                source_info
                            )

                    # ==================================================
                    # PRINT SOURCES
                    # ==================================================

                    print(
                        "\n=============================="
                    )

                    print(
                        "SOURCES"
                    )

                    print(
                        "=============================="
                    )

                    for source, page in sources:

                        print(
                            f"{source} - Page {page}"
                        )

                    # ==================================================
                    # CREATE GEMINI LLM
                    # ==================================================

                    llm = ChatGoogleGenerativeAI(

                        model="gemini-3.6-flash"

                    )

                    # ==================================================
                    # RAG PROMPT
                    # ==================================================

                    prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the information
provided in the document context below.

Do not use outside knowledge.

If the answer cannot be found in the context,
say exactly:

"The information is not available in the document."

Keep the answer clear and concise.

Document Context:

{context}

User Question:

{question}

Answer:
"""

                    # ==================================================
                    # ASK GEMINI
                    # ==================================================

                    response = llm.invoke(
                        prompt
                    )

                    # ==================================================
                    # HANDLE GEMINI RESPONSE
                    # ==================================================

                    if isinstance(
                        response.content,
                        list
                    ):

                        answer = "".join(

                            item.get(
                                "text",
                                ""
                            )

                            for item in response.content

                            if isinstance(
                                item,
                                dict
                            )

                            and item.get(
                                "type"
                            ) == "text"

                        )

                    else:

                        answer = response.content

                    # ==================================================
                    # PRINT AI ANSWER
                    # ==================================================

                    print(
                        "\n=============================="
                    )

                    print(
                        "AI ANSWER"
                    )

                    print(
                        "=============================="
                    )

                    print(
                        answer
                    )

    # ==================================================
    # RENDER PAGE
    # ==================================================

    return render(

        request,

        "qa/index.html",

        {
            "message": message,

            "answer": answer,

            "filename": original_filename,

            "sources": sources
        }

    )