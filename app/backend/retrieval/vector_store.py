import os
import pickle
import logging
from typing import List, Dict, Optional, Any

import chromadb
import bm25s
import Stemmer

logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "papers",
        distance_metric: str = "cosine",  # "cosine", "l2", "ip"
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )
        logger.info(f"ChromaStore: {collection_name} @ {persist_dir} ({self.collection.count()} docs)")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ):
        if not ids:
            return
        
        if metadatas is None:
            metadatas = [{} for _ in ids]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(ids)} docs to ChromaDB")
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        out = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i]
                score = 1 - dist if self.distance_metric == "cosine" else -dist
                out.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score,
                    "source": "semantic",
                })
        return out
    
    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        results = self.collection.get(ids=ids, include=["documents", "metadatas"])
        out = []
        for i, doc_id in enumerate(results["ids"]):
            out.append({
                "id": doc_id,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return out
    
    def count(self) -> int:
        return self.collection.count()
    
    def delete(self, ids: List[str]):
        self.collection.delete(ids=ids)
    
    def reset(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        logger.info(f"Reset collection: {self.collection_name}")


class BM25Store:
    def __init__(self, persist_path: str = "./data/bm25_index.pkl"):
        self.persist_path = persist_path
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = None
        self.corpus = []      # Original texts
        self.ids = []         # Document IDs
        self.metadatas = []   # Metadata for each doc
    
    def add(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ):
        if not ids:
            return
        
        if metadatas is None:
            metadatas = [{} for _ in ids]
        
        self.ids.extend(ids)
        self.corpus.extend(texts)
        self.metadatas.extend(metadatas)
        
        self._build_index()
        logger.info(f"BM25 indexed {len(self.corpus)} docs")
    
    def _build_index(self):
        if not self.corpus:
            return
        
        corpus_tokens = bm25s.tokenize(
            self.corpus,
            stopwords="en",
            stemmer=self.stemmer
        )
        self.retriever = bm25s.BM25(corpus=self.corpus)
        self.retriever.index(corpus_tokens)
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_fn=None,  # Optional: lambda metadata -> bool
    ) -> List[Dict]:
        if self.retriever is None or not self.corpus:
            return []
        
        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=self.stemmer
        )
        
        fetch_k = k * 3 if filter_fn else k
        results = self.retriever.retrieve(query_tokens, k=min(fetch_k, len(self.corpus)))
        
        docs = results.documents.tolist()[0]
        scores = results.scores.tolist()[0]
        
        out = []
        for doc_text, score in zip(docs, scores):
            try:
                idx = self.corpus.index(doc_text)
            except ValueError:
                continue
            
            metadata = self.metadatas[idx]
            
            if filter_fn and not filter_fn(metadata):
                continue
            
            out.append({
                "id": self.ids[idx],
                "text": doc_text,
                "metadata": metadata,
                "score": float(score),
                "source": "bm25",
            })
            
            if len(out) >= k:
                break
        
        return out
    
    def save(self):
        data = {
            "ids": self.ids,
            "corpus": self.corpus,
            "metadatas": self.metadatas,
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 saved to {self.persist_path}")
    
    def load(self) -> bool:
        if not os.path.exists(self.persist_path):
            return False
        
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        
        self.ids = data["ids"]
        self.corpus = data["corpus"]
        self.metadatas = data["metadatas"]
        self._build_index()
        logger.info(f"BM25 loaded: {len(self.corpus)} docs")
        return True
    
    def count(self) -> int:
        return len(self.corpus)
    
    def reset(self):
        self.retriever = None
        self.corpus = []
        self.ids = []
        self.metadatas = []
        if os.path.exists(self.persist_path):
            os.remove(self.persist_path)


class VectorStore:
    def __init__(
        self,
        chroma_dir: str = "./data/chroma",
        bm25_path: str = "./data/bm25_index.pkl",
        collection_name: str = "papers",
    ):
        self.chroma = ChromaStore(chroma_dir, collection_name)
        self.bm25 = BM25Store(bm25_path)
    
    def add(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
    ):
        self.chroma.add(ids, embeddings, texts, metadatas)
        self.bm25.add(ids, texts, metadatas)
    
    def save(self):
        self.bm25.save()
    
    def load(self):
        self.bm25.load()
    
    def count(self) -> Dict[str, int]:
        return {"chroma": self.chroma.count(), "bm25": self.bm25.count()}
    
    def reset(self):
        self.chroma.reset()
        self.bm25.reset()