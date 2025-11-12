class SimpleVectorStore:
    def __init__(self, documents, embeddings_model):
        self.documents = documents
        self.embeddings_model = embeddings_model
        self.embeddings = []
        self._create_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings for all documents"""
        for doc in self.documents:
            # Simulate embedding creation (in real implementation, would use embeddings_model)
            self.embeddings.append(hash(doc.page_content) % 1000)
    
    def similarity_search(self, query, k=3):
        """Simulate similarity search"""
        if not query:  # Return all docs if empty query
            return self.documents[:k]
        
        # Simulate query embedding
        query_embedding = hash(query) % 1000
        
        # Calculate similarities (simplified)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = 1.0 / (1.0 + abs(query_embedding - doc_embedding))
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        return [self.documents[i] for _, i in similarities[:k]]
    
    def as_retriever(self, search_type="similarity", search_kwargs=None):
        """Return a retriever object"""
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        
        class Retriever:
            def __init__(self, vectorstore, search_type, search_kwargs):
                self.vectorstore = vectorstore
                self.search_type = search_type
                self.search_kwargs = search_kwargs
            
            def get_relevant_documents(self, query):
                k = self.search_kwargs.get("k", 3)
                if self.search_type == "mmr":
                    # MMR: Return diverse documents (every other one)
                    docs = self.vectorstore.similarity_search(query, k*2)
                    return docs[::2][:k]
                else:
                    # Similarity: Return most similar
                    return self.vectorstore.similarity_search(query, k)
        
        return Retriever(self, search_type, search_kwargs)