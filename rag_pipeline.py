import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from resume_schema import ResumeSchema
from simple_vectorstore import SimpleVectorStore

load_dotenv()

class ResumeRAG:
    def __init__(self):
        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.parser = PydanticOutputParser(pydantic_object=ResumeSchema)
        self.vectorstore = None
    
    def build_pipeline(self, file_paths):
        """Complete RAG pipeline: load → split → embed → store"""
        # Load
        docs = []
        for path in file_paths:
            docs.extend(TextLoader(path).load())
        
        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Embed & Store
        self.vectorstore = SimpleVectorStore(chunks, self.embeddings)
    
    def extract_structured_resume(self):
        """Schema-based resume extraction"""
        docs = self.vectorstore.similarity_search("", k=10)  # Get all chunks
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = PromptTemplate(
            template="""Extract resume data from context:
{context}

{format_instructions}

Answer:""",
            input_variables=["context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        result = self.llm.invoke(prompt.format(context=context))
        try:
            return self.parser.parse(result)
        except:
            return result
    
    def answer_question(self, question, retrieval_type="similarity"):
        """Answer with different retrieval strategies"""
        if retrieval_type == "similarity":
            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        else:  # MMR
            retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        return self.llm.invoke(prompt)

if __name__ == "__main__":
    rag = ResumeRAG()
    
    # Build pipeline
    rag.build_pipeline(["resumes/sample_resume.txt"])
    
    # Schema-based extraction
    structured_resume = rag.extract_structured_resume()
    print("Structured Resume:", structured_resume)
    
    # Compare retrieval strategies
    question = "What programming languages does the candidate know?"
    print(f"\nSimilarity: {rag.answer_question(question, 'similarity')}")
    print(f"MMR: {rag.answer_question(question, 'mmr')}")