import numpy as np

class similarity_consistency():
    def __init__(self, rg_obj):
        self.rg_obj = rg_obj

    # Get cosine similarity to compare embeddings
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two embedding vectors.

        Parameters:
            a (np.ndarray): The first embedding vector.
            b (np.ndarray): The second embedding vector.

        Returns:
            float: The cosine similarity score between the two vectors.

        Raises:
            ValueError: If the input vectors do not have the same dimension or are empty.

        Example:
            similarity = cosine_similarity(embedding1, embedding2)
        """
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise ValueError("Both 'a' and 'b' must be numpy arrays.")
        if a.size == 0 or b.size == 0:
            raise ValueError("Input vectors must not be empty.")
        if a.shape != b.shape:
            raise ValueError("Input vectors must have the same dimensions.")
        
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    
    def compare_embs(response_embedding, question, rg_obj):
        tmp_metrics = []
        for i in range(2):
            step_response = rg_obj.get_response(question)
            step_embedding = rg_obj.get_embedding(step_response)
            similarity_consistency.cosine_similarity(response_embedding, step_embedding)



    def eval(self, eval_dataset):
        for i in eval_dataset:
            response_embedding = self.rg_obj.embed_query(i.response)



class clarity():
    def __init__(self, name):
        self.name = name

    def compute(self, *args, **kwargs):
        raise NotImplementedError("PENDING")#TODO
