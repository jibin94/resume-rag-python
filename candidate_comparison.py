from rag_pipeline import ResumeRAG

def main():
    print("=== Multi-Resume RAG Pipeline Demo ===\n")
    
    # Initialize RAG system
    rag = ResumeRAG()
    
    # Build pipeline with multiple resumes
    print("Building pipeline with multiple resumes...")
    rag.build_pipeline(["resumes/sample_resume.txt", "resumes/sample_resume2.txt"])
    print("Pipeline built with 2 resumes\n")
    
    # Questions that can be answered from multiple resumes
    questions = [
        "Which candidates have Python experience?",
        "Who has more years of experience?", 
        "What different industries are represented?",
        "Which candidate has cloud platform experience?",
        "Compare the educational backgrounds of the candidates"
    ]
    
    print("Querying across multiple resumes:\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        
        # Compare retrieval strategies
        similarity_answer = rag.answer_question(question, "similarity")
        mmr_answer = rag.answer_question(question, "mmr")
        
        print(f"Similarity: {similarity_answer}")
        print(f"MMR: {mmr_answer}")
        print("-" * 80)
    
    print("\n=== Multi-Resume Analysis Complete ===")

if __name__ == "__main__":
    main()