# Resume RAG Pipeline with LangChain & Google Gemini

Complete implementation of a RAG (Retrieval-Augmented Generation) pipeline for resume processing using LangChain and Google Gemini API.

## Features

**Schema-based Output**: Pydantic models enforce structured resume extraction  
**Complete RAG Pipeline**: load → split → embed → retrieve → answer  
**Retriever Comparison**: Similarity vs MMR retrieval strategies  
**Google Gemini Integration**: Uses gemini-2.5-flash and embedding-001  
**Multi-document Support**: Process multiple resumes simultaneously  
**Vector Search**: FAISS-powered similarity search for efficient retrieval

## Workflow Overview

The RAG pipeline follows a systematic approach to process and analyze resumes:

### 1. Document Ingestion
- Load resume text files from the `resumes/` directory
- Support for multiple document formats and batch processing
- Text preprocessing and cleaning

### 2. Text Chunking
- Split documents into manageable chunks using RecursiveCharacterTextSplitter
- Maintains semantic coherence while ensuring optimal chunk sizes
- Configurable chunk size and overlap parameters

### 3. Embedding Generation
- Convert text chunks into vector representations using Google's embedding-001 model
- High-dimensional embeddings capture semantic meaning
- Consistent embedding space for similarity calculations

### 4. Vector Storage
- Store embeddings in FAISS vector database for efficient retrieval
- Index optimization for fast similarity search
- Support for both similarity and MMR retrieval strategies

### 5. Query Processing
- Accept natural language questions about resume content
- Retrieve relevant document chunks based on semantic similarity
- Rank and filter results using different retrieval algorithms

### 6. Answer Generation
- Use Google Gemini 2.5 Flash to generate contextual responses
- Combine retrieved chunks with user queries
- Structured output using Pydantic schemas for consistent formatting

## Dependencies

The project uses the following key dependencies (see `requirements.txt`):

### Core LangChain Components
- **`langchain-google-genai`** (Latest)
  - Provides Google Gemini AI integration for LangChain framework
  - Includes ChatGoogleGenerativeAI for conversational AI
  - GoogleGenerativeAIEmbeddings for text vectorization
  - Handles API authentication and request management

- **`langchain-community`** (Latest)
  - Community-contributed LangChain components and integrations
  - Additional document loaders, retrievers, and utilities
  - Extended functionality beyond core LangChain features
  - Support for various data sources and formats

- **`langchain-text-splitters`** (Latest)
  - Specialized text splitting utilities for document chunking
  - RecursiveCharacterTextSplitter for intelligent text segmentation
  - Maintains context while creating manageable chunk sizes
  - Configurable splitting strategies and overlap handling

### Vector Storage & Search
- **`faiss-cpu`** (Latest)
  - Facebook AI Similarity Search library for efficient vector operations
  - CPU-optimized version for local development and deployment
  - High-performance similarity search and clustering algorithms
  - Supports various distance metrics and indexing strategies
  - Memory-efficient storage and retrieval of high-dimensional vectors

### Data Validation & Configuration
- **`pydantic`** (Latest)
  - Data validation and schema enforcement using Python type annotations
  - Automatic parsing and validation of structured resume data
  - Type safety and error handling for API responses
  - JSON schema generation for consistent data formats
  - Runtime validation of extracted resume information

- **`python-dotenv`** (Latest)
  - Load environment variables from .env files
  - Secure API key management and configuration
  - Development/production environment separation
  - Simple configuration management without hardcoded secrets  

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure API Key**
```bash
# Add your Google API key to .env
GOOGLE_API_KEY=your_actual_api_key_here
```

3. **Get API Key**
- Visit: https://makersuite.google.com/app/apikey
- Create new API key
- Add to `.env` file

## Usage

### Single Resume Analysis
```bash
python3 single_resume_analysis.py
```

### Candidate Comparison
```bash
python3 candidate_comparison.py
```

### Direct Pipeline Usage
```python
from rag_pipeline import ResumeRAG

rag = ResumeRAG()
rag.build_pipeline(["resumes/resume.txt"])

# Schema extraction
structured_data = rag.extract_structured_resume()

# Q&A with different retrievers
answer = rag.answer_question("What skills does the candidate have?", "similarity")
```

## Pipeline Architecture

```
Documents → TextLoader → RecursiveCharacterTextSplitter → GoogleGenerativeAIEmbeddings → FAISS VectorStore → Retrievers → GoogleGenerativeAI
```

### Detailed Flow
1. **Input**: Resume text files (.txt format)
2. **Loading**: LangChain TextLoader reads documents
3. **Splitting**: RecursiveCharacterTextSplitter creates semantic chunks
4. **Embedding**: Google embedding-001 converts text to vectors
5. **Storage**: FAISS stores vectors for efficient similarity search
6. **Retrieval**: Similarity/MMR retrievers find relevant chunks
7. **Generation**: Gemini 2.5 Flash generates structured responses

## Files

- `rag_pipeline.py` - Main RAG implementation
- `resume_schema.py` - Pydantic schema for structured output
- `simple_vectorstore.py` - Custom vector store (avoids numpy issues)
- `single_resume_analysis.py` - Single resume processing demo
- `candidate_comparison.py` - Multi-candidate comparison demo
- `resumes/sample_resume.txt` - Test resume 1
- `resumes/sample_resume2.txt` - Test resume 2

## Retriever Comparison

- **Similarity**: Returns most semantically similar chunks
- **MMR**: Returns diverse chunks to avoid redundancy

## Requirements Satisfied

1. LangChain schema-based output enforcement
2. Complete RAG pipeline implementation  
3. Similarity vs MMR retriever experimentation
4. Google Gemini API integration

## Output Log Example

![Single Resume Output](https://github.com/jibin94/resume-rag-python/blob/main/screenshot/single_resume1.png)
