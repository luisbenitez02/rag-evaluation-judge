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
    
    
    def compare_embs(expected_response_embedding, question, rg_obj):
        tmp_metrics = []
        for i in range(3):
            #print(i)
            step_response = rg_obj.get_simple_response(question)
            step_embedding = rg_obj.get_embedding(step_response)
            similarity = similarity_consistency.cosine_similarity(np.array(expected_response_embedding), np.array(step_embedding))
            tmp_metrics.append(similarity)
        avg_similarity = np.mean(tmp_metrics)
        #print(f"--- Average similarity: {avg_similarity}")
        return avg_similarity



    def eval(self, eval_dataset):
        similarity_metrics = []
        for i in eval_dataset:
            #print(f"Evaluating similarity consistency of {i}")
            expected_response_embedding = self.rg_obj.get_embedding(i['response'])
            avg_simnilarity = similarity_consistency.compare_embs(expected_response_embedding, i['user_input'], self.rg_obj)
            similarity_metrics.append(avg_simnilarity)
        return round(np.mean(similarity_metrics),4)



class clarity_coherence():
    def __init__(self, judge_agent, rg_obj, default_time=180):
        self.judge_agent = judge_agent
        self.rg_obj = rg_obj
        self.DEFAULT_WAIT_TIME = default_time  # Default wait time for the judge agent to respond
    
    def call_judge_agent(messages,judge_agent):
        ai_msg = judge_agent.llm.invoke(messages)
        return int(ai_msg.content)

    # Evaluate the clarity and coherence of an LLM answer.
    def evaluate_clarity_coherence(self, question: str, expected_answer: str, real_answer: str) -> int:
        CLARITY_COHERENCE_SYSTEM = """You are a writing evaluator. Your task is to evaluate the clarity and coherence of the provided answer. Assess how well-structured and easy to understand the answer is.
        Consider the organization of thoughts, sentence structure, and overall readability.
        Clarity helps the user grasp the content quickly. Compare the provided answer with an expert response to gauge its clarity and coherence.Evaluate with an integer score from 1 to 5, where 1 means "the writing is poor and incoherent" and 5 means "the writing is very clear and entirely coherent, equal or even better than the expert response."
        Return only a single integer score of the overall evaluation result."""

        CLARITY_COHERENCE_USER = f"""Given the question: <<<{question}>>> and the expert response: <<<{expected_answer}>>> evaluate the clarity and coherence of the following answer: <<<{real_answer}>>>.
        Use the following scoring criteria from 1 to 5:
        1 - The writing is poor and incoherent, very difficult to understand.
        2 - The writing is somewhat unclear or disorganized, with frequent issues in clarity or coherence.
        3 - The writing is understandable but has noticeable issues in clarity, structure, or coherence.
        4 - The writing is mostly clear and coherent, with only minor issues.
        5 - The writing is very clear and entirely coherent, equal or even better than the expert response.
        Return only a single integer score of the overall evaluation result.
        """


        if not all(isinstance(arg, str) for arg in [question, expected_answer, real_answer]):
            raise ValueError("All inputs must be strings.")
        
        try:    
            messages = [
                {"role": "system", "content": CLARITY_COHERENCE_SYSTEM},
                {"role": "user", "content": CLARITY_COHERENCE_USER},
            ]
            eval_result = clarity_coherence.call_judge_agent(messages, self.DEFAULT_WAIT_TIME)
            eval_score = int(eval_result)
            print(eval_score)
            return eval_score
        except Exception as e:
            print(f"An error occurred during clarity and coherence evaluation: {str(e)}")
            return 0  # Default score when error occurs
    

    def eval(self, eval_dataset):
        clarity_coherence_metrics = []
        for i in eval_dataset:
            #print(f"Evaluating clarity and coherence of {i}")
            real_answer = self.rg_obj.get_simple_response(i['user_input'])
            eval_score = self.evaluate_clarity_coherence(i['user_input'], i['response'], real_answer)
            clarity_coherence_metrics.append({"user_input": i['user_input'], "score": eval_score})
        
        # Calcular el promedio de los scores
        if len(clarity_coherence_metrics) > 0:
            avg_score = np.mean([item["score"] for item in clarity_coherence_metrics])
            return round(avg_score, 4), clarity_coherence_metrics
        else:
            print("Error getting clarity and coherence metrics.")
            return 0, None
