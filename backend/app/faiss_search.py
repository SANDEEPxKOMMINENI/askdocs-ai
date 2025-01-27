import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional

class FaissSearch:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Use IVFFlat index for better performance with larger document sets
        self.quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(self.quantizer, dimension, 100)
        self.index.train(np.random.rand(1000, dimension).astype(np.float32))
        self.documents = []
        self.content_map: Dict[str, str] = {}
        self.metadata_map: Dict[str, Dict] = {}

    def add_document(self, doc_id: str, content: str, embedding: List[float], metadata: Dict = None):
        if not self.index.is_trained:
            self.index.train(np.array([embedding], dtype=np.float32))
        
        self.index.add(np.array([embedding], dtype=np.float32))
        self.documents.append(doc_id)
        self.content_map[doc_id] = content
        if metadata:
            self.metadata_map[doc_id] = metadata

    def search(self, query_embedding: List[float], k: int = 3, filter_ids: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        query_vector = np.array([query_embedding], dtype=np.float32)
        self.index.nprobe = 10  # Increase search accuracy
        
        distances, indices = self.index.search(query_vector, k * 2)  # Get more results for filtering
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # Valid index
                doc_id = self.documents[idx]
                
                # Apply document filtering
                if filter_ids and doc_id not in filter_ids:
                    continue
                
                content = self.content_map.get(doc_id, "")
                results.append((doc_id, content, float(distance)))
                
                if len(results) >= k:
                    break
        
        return results[:k]

    def get_document_metadata(self, doc_id: str) -> Dict:
        return self.metadata_map.get(doc_id, {})

search_index = FaissSearch()