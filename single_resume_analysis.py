from rag_pipeline import ResumeRAG
import json

def main():
    print("=== Resume RAG Pipeline Demo ===\n")
    
    # Initialize RAG system
    rag = ResumeRAG()
    
    # Build pipeline with resume documents
    print("1. Building RAG Pipeline...")
    print("   - Loading documents")
    print("   - Splitting into chunks") 
    print("   - Creating embeddings")
    print("   - Storing in vector database")
    
    rag.build_pipeline(["resumes/sample_resume.txt"])
    print("   ✅ Pipeline built successfully\n")
    
    # Schema-based extraction
    print("2. Schema-based Resume Extraction:")
    print("   Extracting structured data using Pydantic schema...")
    
    structured_resume = rag.extract_structured_resume()
    
    if hasattr(structured_resume, 'name'):
        print(f"Name: {structured_resume.name}")
        print(f"Email: {structured_resume.email}")
        print(f"Phone: {structured_resume.phone}")
        print(f"Skills: {len(structured_resume.skills)} skills")
        print(f"Experience: {len(structured_resume.experience)} positions")
        print(f"Education: {len(structured_resume.education)} degrees")
        print("Structured extraction successful\n")
    else:
        print(f"Raw output: {structured_resume}\n")
    
    # Question answering with different retrievers
    print("3. Retriever Comparison:")
    
    questions = [
        "What programming languages does the candidate know?",
        "How many years of experience does the candidate have?",
        "What companies has the candidate worked for?",
        "What is the candidate's educational background?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n   Question {i}: {question}")
        
        # Similarity retrieval
        similarity_answer = rag.answer_question(question, "similarity")
        print(f"Similarity: {similarity_answer}")
        
        # MMR retrieval  
        mmr_answer = rag.answer_question(question, "mmr")
        print(f"MMR: {mmr_answer}")
    
    print("\n=== Demo Complete ===")
    print("\nPipeline Components Used:")
    print("LangChain TextLoader (Load)")
    print("RecursiveCharacterTextSplitter (Split)")
    print("GoogleGenerativeAIEmbeddings (Embed)")
    print("SimpleVectorStore (Store & Retrieve)")
    print("GoogleGenerativeAI (Answer)")
    print("PydanticOutputParser (Schema)")
    print("Similarity vs MMR Retrievers")

if __name__ == "__main__":
    main()