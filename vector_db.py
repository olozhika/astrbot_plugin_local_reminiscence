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
        import torch
        
        # 1. 探测系统默认路径 (System Default Cache)
        # 通常是 ~/.cache/torch/sentence_transformers/
        try:
            default_torch_home = torch.hub._get_torch_home()
            default_st_home = Path(default_torch_home) / 'sentence_transformers'
        except Exception:
            default_st_home = Path.home() / '.cache' / 'torch' / 'sentence_transformers'
        
        # 2. 探测插件自定义缓存路径 (Custom Cache Dir)
        # 按照用户要求，如果没设置 model_cache_dir，我们也可以尝试从 db_path (即 vector_db_path) 找
        custom_cache_dir = None
        if model_cache_dir:
            custom_cache_dir = Path(model_cache_dir).expanduser().resolve()
        else:
            # 如果没设置模型缓存目录，尝试使用向量数据库目录作为备选
            custom_cache_dir = Path(self.db_path).parent / "APLR_ModelCache"
            
        # 确保自定义目录存在
        custom_cache_dir.mkdir(parents=True, exist_ok=True)

        # 3. 构建候选路径列表 (Candidate Paths)
        # 这里的逻辑是：如果 model_name 是个 ID，我们先去这些地方找找看有没有现成的文件夹
        candidate_names = []
        
        # A. 检查是否是绝对路径或当前目录下存在的路径
        if Path(model_name).exists():
            candidate_names.append(str(Path(model_name).resolve()))

        # B. 检查系统默认路径是否存在该模型
        # 转换模型名为文件夹名 (通常是下划线替换斜杠)
        safe_model_name = model_name.replace("/", "_")
        if (default_st_home / safe_model_name).exists():
            candidate_names.append(str(default_st_home / safe_model_name))
        if (default_st_home / model_name).exists():
            candidate_names.append(str(default_st_home / model_name))

        # C. 检查自定义缓存路径是否存在该模型
        if (custom_cache_dir / safe_model_name).exists():
            candidate_names.append(str(custom_cache_dir / safe_model_name))
        if (custom_cache_dir / model_name).exists():
            candidate_names.append(str(custom_cache_dir / model_name))

        # 4. 如果没找到现成的，我们准备下载。
        # 在下载前，我们需要设置环境变量，告诉 SDK 往哪儿下。
        # 按照用户要求，如果本地都没有，下载到 vector_db_path 相关的目录
        # 这里我们统一使用 custom_cache_dir (它通常是 APLR_ModelCache)
        # 如果你希望严格下载到 APLR_VectorDB 内部，可以将 custom_cache_dir 设为 self.db_path
        
        os.environ.setdefault("HF_HOME", str(custom_cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(custom_cache_dir / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(custom_cache_dir / "transformers"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(custom_cache_dir / "sentence_transformers"))

        if hf_endpoint and not offline_mode:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)

        model_kwargs = {
            "trust_remote_code": trust_remote_code
        }
        if custom_cache_dir:
            model_kwargs["cache_folder"] = str(custom_cache_dir)
        if offline_mode:
            model_kwargs["local_files_only"] = True

        def _build_model(name: str) -> SentenceTransformer:
            return SentenceTransformer(name, **model_kwargs)

        def _reset_hf_http_session():
            try:
                from huggingface_hub.utils import _http as hf_http  # type: ignore
                reset_fn = getattr(hf_http, "reset_sessions", None)
                if callable(reset_fn):
                    reset_fn()
            except Exception:
                pass

        # 5. 尝试加载
        # 先试候选路径（本地已有的）
        for candidate in candidate_names:
            try:
                return _build_model(candidate)
            except Exception:
                continue

        # 如果候选路径都失败了，最后尝试用原始名称加载（这可能会触发下载）
        try:
            return _build_model(model_name)
        except Exception as exc:
            if "client has been closed" in str(exc).lower():
                _reset_hf_http_session()
                try:
                    return _build_model(model_name)
                except Exception as exc2:
                    exc = exc2
            
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
