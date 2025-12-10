from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def merge_results(
    semantic_results: List[Dict],
    bm25_results: List[Dict],
    k: int,
) -> List[Dict]:
    half_k = k // 2
    seen_ids = set()
    merged = []
    
    for doc in semantic_results[:half_k]:
        if doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            merged.append(doc)
    
    for doc in bm25_results[:half_k]:
        if doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            merged.append(doc)
    
    for doc in semantic_results[half_k:] + bm25_results[half_k:]:
        if len(merged) >= k:
            break
        if doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            merged.append(doc)
    
    return merged

class HybridSearcher:
    def __init__(
        self,
        vector_store,   
        embedding_model,  
        reranker=None, 
    ):
        self.store = vector_store
        self.embedder = embedding_model
        self.reranker = reranker
    
    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 20,
        use_rerank: bool = True,
        where: Optional[Dict] = None,
    ) -> Dict:
        half_k = initial_k // 2
        
        query_embedding = self.embedder.embed_query_sync(query)
        semantic_results = self.store.chroma.search(query_embedding, k=half_k, where=where)
        bm25_results = self.store.bm25.search(query, k=half_k)
        merged = merge_results(semantic_results, bm25_results, initial_k)
        
        if use_rerank and self.reranker and merged:
            docs_to_rerank = [r["text"] for r in merged]
            reranked = self.reranker.rerank_sync(query, docs_to_rerank, top_k=top_k)
            
            text_to_doc = {d["text"]: d for d in merged}
            results = []
            for r in reranked:
                doc = text_to_doc[r["text"]].copy()
                doc["rerank_score"] = r["score"]
                results.append(doc)
        else:
            results = merged[:top_k]
        
        return {
            "results": results,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "query": query,
        }
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 20,
        use_rerank: bool = True,
        where: Optional[Dict] = None,
    ) -> Dict:
        half_k = initial_k // 2
        
        query_embedding = await self.embedder.embed_query(query)
        semantic_results = self.store.chroma.search(query_embedding, k=half_k, where=where)
        bm25_results = self.store.bm25.search(query, k=half_k)
        merged = merge_results(semantic_results, bm25_results, initial_k)
        
        if use_rerank and self.reranker and merged:
            docs_to_rerank = [r["text"] for r in merged]
            reranked = await self.reranker.rerank(query, docs_to_rerank, top_k=top_k)
            
            text_to_doc = {d["text"]: d for d in merged}
            results = []
            for r in reranked:
                doc = text_to_doc[r["text"]].copy()
                doc["rerank_score"] = r["score"]
                results.append(doc)
        else:
            results = merged[:top_k]
        
        return {
            "results": results,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "query": query,
        }
    
    def semantic_only_sync(self, query: str, k: int = 10, where: Dict = None) -> List[Dict]:
        emb = self.embedder.embed_query_sync(query)
        return self.store.chroma.search(emb, k=k, where=where)
    
    def bm25_only_sync(self, query: str, k: int = 10) -> List[Dict]:
        return self.store.bm25.search(query, k=k)