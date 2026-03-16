import chromadb
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict

class VectorDB:
    def __init__(self, db_path: str, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', offline_mode: bool = False):
        self.db_path = db_path
        
        if offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else: 
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
            
        # 显式配置 Settings，增加一些稳定性
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self.client.get_or_create_collection(name="events")
        self.model = SentenceTransformer(model_name,local_files_only=offline_mode)

    def add_events(self, events: List[Dict]):
        """将事件向量化并存入 ChromaDB"""
        if not events:
            return
        
        ids = [ev['event_id'] for ev in events]
        documents = [ev['narrative'] for ev in events]
        # 开启归一化
        embeddings = self.model.encode(documents, normalize_embeddings=True).tolist()
        
        # ChromaDB upsert
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents
        )

    def search_events(self, query: str, top_n: int = 10) -> List[Dict]:
        """搜索最接近的事件 ID 和相关度分数"""
        # 开启归一化
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_n
        )
        
        output = []
        if results['ids'] and results['distances']:
            ids = results['ids'][0]
            distances = results['distances'][0]
            for eid, dist in zip(ids, distances):
                # 当向量已归一化时，ChromaDB 的 L2 距离 d 与余弦相似度 sim 的关系为：d^2 = 2(1 - sim)
                # 因此 sim = 1 - (d^2 / 2)
                # 注意：ChromaDB 返回的可能是 d^2
                cosine_sim = 1 - (dist / 2)
                relevance = max(0, cosine_sim) * 100
                
                output.append({
                    "event_id": eid,
                    "relevance": round(relevance, 1)
                })
        return output
