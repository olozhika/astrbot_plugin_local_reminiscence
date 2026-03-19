import os
from pathlib import Path
from typing import List, Dict, Optional

class VectorDB:
    def __init__(
        self,
        db_path: str,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        model_cache_dir: Optional[str] = None,
        hf_endpoint: str = "",
        trust_remote_code: bool = False,
        offline_mode: bool = False
    ):
        self.db_path = db_path
        self.offline_mode = offline_mode
        
        if self.offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else: 
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
            # 如果没有提供镜像地址，确保清除环境变量以便回退到官方地址
            if not hf_endpoint:
                os.environ.pop("HF_ENDPOINT", None)
            
        # 延迟导入 chromadb 以确保环境变量已设置
        import chromadb
        # 显式配置 Settings，增加一些稳定性
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self.client.get_or_create_collection(name="events")
        
        self.model = None
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.hf_endpoint = hf_endpoint
        self.trust_remote_code = trust_remote_code

    def _load_embedding_model(
        self,
        model_name: str,
        model_cache_dir: Optional[str],
        hf_endpoint: str,
        trust_remote_code: bool,
        offline_mode: bool
    ):
        # 延迟导入以确保环境变量已设置
        from sentence_transformers import SentenceTransformer
        
        cache_dir = None
        if model_cache_dir:
            cache_dir = str(Path(model_cache_dir).expanduser().resolve())
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path / "transformers"))
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_path / "sentence_transformers"))

        if hf_endpoint and not offline_mode:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)

        model_kwargs = {
            "trust_remote_code": trust_remote_code
        }
        if cache_dir:
            model_kwargs["cache_folder"] = cache_dir
        if offline_mode:
            model_kwargs["local_files_only"] = True

        def _build_model(name: str) -> SentenceTransformer:
            return SentenceTransformer(name, **model_kwargs)

        def _reset_hf_http_session():
            # 某些 huggingface_hub/httpx 版本可能出现 "client has been closed"
            # 尝试重置全局会话后重试一次
            try:
                from huggingface_hub.utils import _http as hf_http  # type: ignore
                reset_fn = getattr(hf_http, "reset_sessions", None)
                if callable(reset_fn):
                    reset_fn()
            except Exception:
                pass

        candidate_names = []
        # 1. 优先检查 model_name 是否为本地存在的绝对或相对路径
        if Path(model_name).exists():
            candidate_names.append(model_name)
        
        # 2. 检查 cache 目录下是否已经存在该模型（针对短名）
        if cache_dir:
            cache_path = Path(cache_dir)
            if (cache_path / model_name).exists():
                candidate_names.append(str(cache_path / model_name))
            if (cache_path / f"sentence-transformers_{model_name}").exists():
                candidate_names.append(str(cache_path / f"sentence-transformers_{model_name}"))
            # 某些版本存储为 sentence-transformers/model_name
            if (cache_path / "sentence-transformers" / model_name).exists():
                candidate_names.append(str(cache_path / "sentence-transformers" / model_name))

        # 3. 添加原始名称和补全名称作为备选（由 SDK 处理下载或缓存查找）
        if model_name not in candidate_names:
            candidate_names.append(model_name)
        
        if ("/" not in model_name) and (not Path(model_name).exists()):
            full_name = f"sentence-transformers/{model_name}"
            if full_name not in candidate_names:
                candidate_names.append(full_name)

        last_exc = None
        for idx, candidate in enumerate(candidate_names):
            try:
                return _build_model(candidate)
            except Exception as exc:
                last_exc = exc
                if "client has been closed" in str(exc).lower():
                    _reset_hf_http_session()
                    try:
                        return _build_model(candidate)
                    except Exception as exc2:
                        last_exc = exc2
                if idx < len(candidate_names) - 1:
                    continue
                break

        try:
            _reset_hf_http_session()
            return _build_model(model_name)
        except Exception as exc:
            tips = [
                f"模型加载失败: {model_name}",
                f"原始错误: {exc}",
                "可选排查：",
                "1) 设置可访问的 HF 镜像地址（如 https://hf-mirror.com）；",
                "2) 把 embedding_model 改成已下载好的本地目录；",
                "3) 若已离线缓存，开启 offline_mode=true。"
            ]
            raise RuntimeError("\n".join(tips)) from exc

    def _ensure_model(self):
        if self.model is None:
            self.model = self._load_embedding_model(
                model_name=self.model_name,
                model_cache_dir=self.model_cache_dir,
                hf_endpoint=self.hf_endpoint,
                trust_remote_code=self.trust_remote_code,
                offline_mode=self.offline_mode
            )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量表示（供外部插件调用）"""
        if not texts:
            return []
        self._ensure_model()
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def add_events(self, events: List[Dict]):
        """将事件向量化并存入 ChromaDB"""
        if not events:
            return
        self._ensure_model()
        
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
        self._ensure_model()
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
