# AI Document Q&A

A Django-based Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural-language questions about their content.

The system extracts text from uploaded PDFs, splits the content into smaller chunks, generates vector embeddings using Google Gemini, stores them in ChromaDB, retrieves the most relevant chunks for a question, and uses Gemini to generate a grounded answer with document source information.

## Features

- Upload PDF documents through a web interface
- Extract text from PDF files
- Split documents into overlapping chunks
- Generate embeddings using Google Gemini
- Store and retrieve document vectors using ChromaDB
- Semantic similarity-based document retrieval
- Retrieve the top relevant chunks for each question
- Similarity threshold to reduce irrelevant context
- Generate answers using Google Gemini
- Ground responses using retrieved document context
- Display document source and page information
- Environment-variable based API key configuration
- Document-specific vector collections

## RAG Architecture

```text
                    ┌──────────────────┐
                    │   Upload PDF     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  PDF Text        │
                    │  Extraction      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Text Chunking    │
                    │ 500 / 100        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Gemini           │
                    │ Embeddings       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   ChromaDB       │
                    │ Vector Store     │
                    └──────────────────┘

User Question
      │
      ▼
┌──────────────────┐
│ Gemini Embedding │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ChromaDB Search  │
│ Top-K Retrieval  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Similarity Filter│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Retrieved Context│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Gemini LLM       │
│ Answer Generation│
└────────┬─────────┘
         │
         ▼
     Final Answer
     + Sources