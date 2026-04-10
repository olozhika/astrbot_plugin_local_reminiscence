import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from astrbot.api import logger

class VectorDB:
    def __init__(
        self,
        db_path: str,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        model_cache_dir: Optional[str] = None,
        hf_endpoint: str = "",
        trust_remote_code: bool = False,
        offline_mode: bool = False,
        ai_name: str = ""
    ):
        self.db_path = db_path
        self.offline_mode = offline_mode
        self.ai_name = ai_name
        
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
        
        # 1. 确定搜索目录
        search_roots = []
        
        # A. 环境变量指定的路径
        for env_var in ["SENTENCE_TRANSFORMERS_HOME", "HF_HOME", "XDG_CACHE_HOME"]:
            val = os.environ.get(env_var)
            if val:
                p = Path(val)
                if env_var == "HF_HOME": p = p / "sentence_transformers"
                if env_var == "XDG_CACHE_HOME": p = p / "torch" / "sentence_transformers"
                search_roots.append(p)

        # B. 系统默认路径
        try:
            default_torch_home = torch.hub._get_torch_home()
            search_roots.append(Path(default_torch_home) / 'sentence_transformers')
        except Exception:
            pass
        search_roots.append(Path.home() / '.cache' / 'torch' / 'sentence_transformers')
        search_roots.append(Path.home() / '.cache' / 'huggingface' / 'sentence_transformers')
        search_roots.append(Path.home() / '.cache' / 'huggingface' / 'hub')
        
        # C. 插件自定义路径
        plugin_cache_dir = Path(model_cache_dir).expanduser().resolve() if model_cache_dir else Path(self.db_path).parent / "APLR_ModelCache"
        search_roots.append(plugin_cache_dir)
        
        # D. 当前目录及模型目录
        search_roots.append(Path.cwd() / "models")
        search_roots.append(Path.cwd())

        # 去重并保留顺序
        unique_roots = []
        for r in search_roots:
            if r not in unique_roots: unique_roots.append(r)
        
        # 2. 定义探测函数：寻找包含 config.json 的有效模型文件夹
        def find_local_path(roots: List[Path], name: str) -> Optional[str]:
            short_name = name.split('/')[-1]
            search_names = [name, name.replace("/", "_"), short_name]
            
            for root in roots:
                if not root.exists(): continue
                # 优先尝试精确匹配
                for sn in search_names:
                    p = root / sn
                    if p.exists() and (p / "config.json").exists():
                        return str(p.resolve())
                    # 尝试前缀匹配 (针对 sentence-transformers_ 这种)
                    p2 = root / f"sentence-transformers_{sn}"
                    if p2.exists() and (p2 / "config.json").exists():
                        return str(p2.resolve())
                
                # 模糊匹配：遍历一级子目录
                try:
                    for p in root.iterdir():
                        if not p.is_dir(): continue
                        if short_name in p.name and (p / "config.json").exists():
                            return str(p.resolve())
                        # 针对 HF Hub 结构 (models--.../snapshots/.../config.json)
                        if "models--" in p.name and short_name in p.name:
                            snapshots = p / "snapshots"
                            if snapshots.exists():
                                for snap in snapshots.iterdir():
                                    if snap.is_dir() and (snap / "config.json").exists():
                                        return str(snap.resolve())
                except Exception:
                    continue
            return None

        # 3. 优先级探测
        candidate_paths = []
        
        # A. 检查是否直接是路径 (绝对或相对)
        p_direct = Path(model_name)
        if p_direct.exists() and (p_direct / "config.json").exists():
            candidate_paths.append(str(p_direct.resolve()))
        
        # B. 探测所有根目录
        found_path = find_local_path(unique_roots, model_name)
        if found_path:
            candidate_paths.append(found_path)

        # 4. 尝试加载本地候选
        for path in candidate_paths:
            try:
                # 显式指定 local_files_only=True 确保不联网
                return SentenceTransformer(path, trust_remote_code=trust_remote_code, local_files_only=True)
            except Exception:
                continue

        # 5. 如果本地没找到，且是离线模式，直接报错
        if offline_mode:
            searched_str = "\n".join([f"- {r}" for r in unique_roots if r.exists()])
            raise RuntimeError(
                f"离线模式下未找到模型 {model_name}。\n"
                f"已搜索根目录:\n{searched_str}\n"
                "请确保模型文件夹（包含 config.json）存在于上述路径中。"
            )

        # 6. 准备在线下载
        # 设置环境变量
        os.environ["HF_HOME"] = str(plugin_cache_dir)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(plugin_cache_dir / "sentence_transformers")
        
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            try:
                import huggingface_hub
                huggingface_hub.constants.HF_ENDPOINT = hf_endpoint
            except Exception:
                pass

        def _reset_hf_http_session():
            try:
                from huggingface_hub.utils import _http as hf_http  # type: ignore
                reset_fn = getattr(hf_http, "reset_sessions", None)
                if callable(reset_fn):
                    reset_fn()
            except Exception:
                pass

        try:
            # 尝试下载并加载，指定 cache_folder 确保下载到插件目录
            return SentenceTransformer(model_name, trust_remote_code=trust_remote_code, cache_folder=str(plugin_cache_dir))
        except Exception as exc:
            if "client has been closed" in str(exc).lower():
                _reset_hf_http_session()
                try:
                    return SentenceTransformer(model_name, trust_remote_code=trust_remote_code, cache_folder=str(plugin_cache_dir))
                except Exception as exc2:
                    exc = exc2
            
            # 收集诊断信息：列出搜索目录下的内容
            diagnostic_info = []
            for r in unique_roots:
                if r.exists():
                    try:
                        dirs = [d.name for d in r.iterdir() if d.is_dir()]
                        diagnostic_info.append(f"目录 {r} 下的文件夹: {dirs[:10]}{'...' if len(dirs)>10 else ''}")
                    except Exception:
                        diagnostic_info.append(f"无法读取目录 {r}")

            tips = [
                f"模型加载失败: {model_name}",
                f"已尝试本地路径: {candidate_paths}",
                f"搜索根目录: {[str(r) for r in unique_roots]}",
                "诊断信息:",
                *diagnostic_info,
                f"原始错误: {exc}",
                "建议：",
                "1) 检查本地模型文件夹是否完整（必须包含 config.json）；",
                "2) 检查网络或镜像地址是否正确；",
                "3) 手动将模型文件夹放入 APLR_ModelCache 目录下。"
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

    def clear_all(self):
        """清空向量库中的所有数据"""
        try:
            self.client.delete_collection(name="events")
            self.collection = self.client.get_or_create_collection(name="events")
            logger.info("[APLR] 向量库已成功清空。")
        except Exception as e:
            logger.error(f"[APLR] 清空向量库失败: {e}")
            raise e

    def add_events(self, events: List[Dict]):
        """将事件向量化并存入 ChromaDB"""
        if not events:
            return
        self._ensure_model()
        
        ids = [ev['event_id'] for ev in events]
        documents = [ev['narrative'] for ev in events]
        
        # 替换 "我" 为 ai_name 用于向量化，但不影响存储的原始文本
        embedding_texts = documents
        if self.ai_name:
            # 使用正则替换 "我"，但排除 "我们"
            # 模式：匹配 "我"，且后面不跟着 "们"，前面不跟着 "自"
            pattern = r'(?<!自)我(?![们])'
            embedding_texts = [re.sub(pattern, self.ai_name, text) for text in documents]
            
        # 开启归一化
        embeddings = self.model.encode(embedding_texts, normalize_embeddings=True).tolist()
        
        # ChromaDB upsert
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents
        )

    def search_events(self, query: str, top_n: int = 10) -> List[Dict]:
        """搜索最接近的事件 ID 和相关度分数"""
        self._ensure_model()
        
        import logging
        logger = logging.getLogger("AstrBot")
        logger.debug(f"[APLR] VectorDB 正在搜索: {query}")
        
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
