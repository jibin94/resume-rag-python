# Resume RAG Pipeline with LangChain & Google Gemini

Complete implementation of a RAG (Retrieval-Augmented Generation) pipeline for resume processing using LangChain and Google Gemini API.

## Features

**Schema-based Output**: Pydantic models enforce structured resume extraction  
**Complete RAG Pipeline**: load → split → embed → retrieve → answer  
**Retriever Comparison**: Similarity vs MMR retrieval strategies  
**Google Gemini Integration**: Uses gemini-2.5-flash and embedding-001  
**Multi-document Support**: Process multiple resumes simultaneously  

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
Documents → TextLoader → RecursiveCharacterTextSplitter → GoogleGenerativeAIEmbeddings → SimpleVectorStore → Retrievers → GoogleGenerativeAI
```

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