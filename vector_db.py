import os
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(
        self,
        db_path: str,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        model_cache_dir: Optional[str] = None,
        hf_endpoint: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False
    ):
        self.db_path = db_path
        # 显式配置 Settings，增加一些稳定性
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self.client.get_or_create_collection(name="events")
        self.model = self._load_embedding_model(
            model_name=model_name,
            model_cache_dir=model_cache_dir,
            hf_endpoint=hf_endpoint,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code
        )

    def _load_embedding_model(
        self,
        model_name: str,
        model_cache_dir: Optional[str],
        hf_endpoint: str,
        local_files_only: bool,
        trust_remote_code: bool
    ) -> SentenceTransformer:
        cache_dir = None
        if model_cache_dir:
            cache_dir = str(Path(model_cache_dir).expanduser().resolve())
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path / "transformers"))
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_path / "sentence_transformers"))

        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        model_kwargs = {
            "trust_remote_code": trust_remote_code
        }
        if cache_dir:
            model_kwargs["cache_folder"] = cache_dir
        if local_files_only:
            model_kwargs["local_files_only"] = True

        try:
            return SentenceTransformer(model_name, **model_kwargs)
        except Exception as exc:
            tips = [
                f"模型加载失败: {model_name}",
                f"原始错误: {exc}",
                "可选排查：",
                "1) 设置可访问的 HF 镜像地址（如 https://hf-mirror.com）；",
                "2) 把 embedding_model 改成已下载好的本地目录；",
                "3) 若已离线缓存，开启 embedding_local_files_only=true。"
            ]
            raise RuntimeError("\n".join(tips)) from exc

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
