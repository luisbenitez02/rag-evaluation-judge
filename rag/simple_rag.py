import numpy as np
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

#This code is a simple implementation of a Retrieval-Augmented Generation (RAG) system using Azure OpenAI services. Originally written by Ragas
#and modified by the author to use Azure OpenAI services for both document embedding and question answering.

class RAG:
    def __init__(self, endpoint_az,model="o4-mini_pv", embedding_model="text-embedding-3-large"):
        self.llm = AzureChatOpenAI(
            azure_endpoint= f'https://{endpoint_az}.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview',#os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=model,#os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version='2024-12-01-preview')#os.environ["AZURE_OPENAI_API_VERSION"])
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint= f'https://{endpoint_az}.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15',#os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=embedding_model,#os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version='2024-02-01')
        self.doc_embeddings = None
        self.docs = None

    def load_documents(self, documents):
        """Load documents and compute their embeddings."""
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query):
        """Find the most relevant document for a given query."""
        if not self.docs or not self.doc_embeddings:
            raise ValueError("Documents and their embeddings are not loaded.")

        query_embedding = self.embeddings.embed_query(query)
        similarities = [
            np.dot(query_embedding, doc_emb)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]

    def generate_answer(self, query, relevant_doc):
        """Generate an answer for a given query based on the most relevant document."""
        prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content