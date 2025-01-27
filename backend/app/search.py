import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import json

class SimpleSearch:
    def search(self, query_embedding: List[float], documents: List[any], k: int = 3) -> List[Tuple[str, str, float]]:
        if not documents:
            return []
            
        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding).reshape(1, -1)
        
        results = []
        for doc in documents:
            doc_embedding = np.array(doc.get_embedding()).reshape(1, -1)
            similarity = float(cosine_similarity(query_vector, doc_embedding)[0][0])
            results.append((doc.id, doc.content, similarity))
        
        # Sort by similarity and return top k
        return sorted(results, key=lambda x: x[2], reverse=True)[:k]

search_engine = SimpleSearch()