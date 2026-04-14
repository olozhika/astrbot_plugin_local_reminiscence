import numpy as np
import jieba
import re
import json
from collections import Counter
from typing import List, Dict, Any, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from astrbot.api import logger

from .database import MemoryDB
from .vector_db import VectorDB
from .models import ThematicMemory

class ThematicConsolidator:
    def __init__(self, db: MemoryDB, vector_db: VectorDB, llm_generate_func, ai_name: str = "", persona: str = "", prompt_theme_summary: str = ""):
        self.db = db
        self.vector_db = vector_db
        self.llm_generate = llm_generate_func
        self.ai_name = ai_name
        self.persona = persona
        self.prompt_theme_summary = prompt_theme_summary

    def _get_pure_narrative(self, text: str) -> str:
        """去除日期前缀 [YYYY-MM-DD]"""
        return re.sub(r'^\[\d{4}-\d{2}-\d{2}\]\s*', '', text)

    def _get_dynamic_keywords(self, texts: List[str], stop_threshold=0.2):
        """动态识别高频词作为停用词，并提取每条文本的关键词"""
        all_words = []
        doc_words = []
        
        for text in texts:
            # 仅保留中文字符、字母和数字进行分词
            clean_text = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text))
            words = [w for w in jieba.lcut(clean_text) if len(w) > 1]
            all_words.extend(words)
            doc_words.append(set(words))
            
        # 计算词频
        word_counts = Counter(all_words)
        total_docs = len(texts)
        
        # 识别在超过 stop_threshold 比例的文档中出现的词
        dynamic_stopwords = set()
        for word, count in word_counts.items():
            doc_count = sum(1 for dw in doc_words if word in dw)
            if doc_count / total_docs > stop_threshold:
                dynamic_stopwords.add(word)
                
        # 提取每条文本的有效关键词
        keywords_per_doc = []
        for dw in doc_words:
            keywords_per_doc.append(dw - dynamic_stopwords)
            
        return keywords_per_doc, word_counts, total_docs, dynamic_stopwords

    def _compute_hybrid_distance_matrix(self, embeddings: np.ndarray, keywords_per_doc: List[set], word_counts: Counter, total_docs: int, boost_factor=0.15):
        """计算结合向量距离和关键词奖励的混合距离矩阵"""
        # 1. 计算基础余弦距离 (0 到 2 之间，0表示完全相同)
        base_distances = pairwise_distances(embeddings, metric='cosine')
        
        # 2. 计算关键词奖励
        n = len(keywords_per_doc)
        keyword_rewards = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                common_keys = keywords_per_doc[i].intersection(keywords_per_doc[j])
                if common_keys:
                    reward = 0
                    for key in common_keys:
                        # TF-IDF 思想：越罕见的词奖励越高
                        freq = word_counts[key] / total_docs
                        weight = 1.0 / (freq + 0.01)
                        reward += weight
                    
                    # 归一化奖励并应用增强系数
                    norm_reward = np.log1p(reward) * boost_factor
                    keyword_rewards[i, j] = keyword_rewards[j, i] = norm_reward
                    
        # 3. 混合距离 = 基础距离 - 奖励 (最小限制为 0)
        hybrid_matrix = np.maximum(0, base_distances - keyword_rewards)
        return hybrid_matrix

    async def big_consolidation(self, target_min=15, target_max=35):
        """全局重聚类（大固化）"""
        logger.info("[APLR] 开始执行全局记忆固化（大固化）...")
        
        # 1. 获取所有事件
        events = self.db.get_all_events()
        if len(events) < target_min:
            logger.info(f"[APLR] 事件数量不足 ({len(events)} < {target_min})，跳过固化。")
            return
            
        event_ids = [e['event_id'] for e in events]
        pure_narratives = [e['narrative'] for e in events]
        
        # 2. 获取向量
        collection = self.vector_db.client.get_collection("events")
        res = collection.get(ids=event_ids, include=['embeddings'])
        
        # 建立 ID 到向量的映射
        id_to_embedding = {res['ids'][i]: res['embeddings'][i] for i in range(len(res['ids']))}
        embeddings = np.array([id_to_embedding[eid] for eid in event_ids])
        embeddings = normalize(embeddings)
        
        # 3. 提取关键词
        keywords_per_doc, word_counts, total_docs, dynamic_stopwords = self._get_dynamic_keywords(pure_narratives)
        
        # 保存全局高频词供增量固化使用
        self.db.set_consolidation_config("global_high_freq_words", json.dumps(list(dynamic_stopwords), ensure_ascii=False))
        
        # 4. 计算混合距离矩阵
        hybrid_matrix = self._compute_hybrid_distance_matrix(embeddings, keywords_per_doc, word_counts, total_docs)
        
        # 5. 自适应阈值搜索
        target_mid = (target_min + target_max) / 2
        best_threshold = 0.5
        best_clusters = None
        best_is_valid = False
        best_diff = float('inf')
        
        for threshold in np.linspace(0.3, 0.8, 11):
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='precomputed',
                linkage='complete'
            )
            labels = clustering.fit_predict(hybrid_matrix)
            
            # 分析结果
            unique_labels, counts = np.unique(labels, return_counts=True)
            valid_themes = sum(1 for c in counts if c >= 4)
            max_cluster_ratio = max(counts) / len(events)
            
            # 检查是否满足基本约束
            is_valid = (target_min <= valid_themes <= target_max) and (max_cluster_ratio < 0.15)
            diff = abs(valid_themes - target_mid)
            
            # 更新逻辑：
            # 1. 如果当前方案有效且之前没有有效方案 -> 选它
            # 2. 如果当前方案有效且之前也有有效方案 -> 选更接近中点的
            # 3. 如果当前方案无效且之前也没有有效方案 -> 选更接近中点的（兜底）
            if best_clusters is None:
                best_threshold = threshold
                best_clusters = labels
                best_is_valid = is_valid
                best_diff = diff
            else:
                if (is_valid and not best_is_valid) or (is_valid == best_is_valid and diff < best_diff):
                    best_threshold = threshold
                    best_clusters = labels
                    best_is_valid = is_valid
                    best_diff = diff

        # 6. 保存结果
        self.db.clear_thematic_memories()
        self.db.set_consolidation_config("last_threshold", str(best_threshold))
        
        # 清空向量库中的旧主题
        theme_collection = self.vector_db.client.get_or_create_collection("themes")
        all_themes = theme_collection.get()
        if all_themes['ids']:
            theme_collection.delete(ids=all_themes['ids'])

        unique_labels = np.unique(best_clusters)
        for label in unique_labels:
            indices = np.where(best_clusters == label)[0]
            if len(indices) < 4:
                continue # 忽略小簇
                
            theme_id = f"theme_{label:03d}"
            cluster_event_ids = [event_ids[i] for i in indices]
            cluster_embeddings = embeddings[indices]
            centroid = np.mean(cluster_embeddings, axis=0).astype(np.float32)
            
            # 提取主题关键词 (仅使用过滤后的关键词，统计文档频率)
            theme_word_counts = Counter()
            for idx in indices:
                # 使用已经剔除了全局高频词的 keywords_per_doc
                theme_word_counts.update(keywords_per_doc[idx])
            
            top_keywords = [w for w, _ in theme_word_counts.most_common(10)]
            
            # 1. 保存到 SQL 数据库 (作为真理来源)
            self.db.save_thematic_memory(
                theme_id=theme_id,
                summary="", # 稍后异步总结
                keywords=json.dumps(top_keywords, ensure_ascii=False),
                event_count=len(cluster_event_ids)
            )
            
            # 2. 保存到向量数据库 (用于快速检索)
            theme_collection.add(
                ids=[theme_id],
                embeddings=[centroid.tolist()],
                metadatas=[{"type": "theme"}]
            )
            
            for eid in cluster_event_ids:
                self.db.map_event_to_theme(eid, theme_id)
                
        logger.info(f"[APLR] 大固化完成。找到 {len(unique_labels)} 个簇，其中有效主题数: {sum(1 for label in unique_labels if sum(best_clusters == label) >= 4)}")
        
        # 7. 触发总结
        await self.summarize_all_themes()

    async def incremental_consolidation(self):
        """增量固化（日常更新）"""
        logger.info("[APLR] 开始执行增量记忆固化...")
        
        # 1. 获取未分类事件
        unthemed_events = self.db.get_unthemed_events()
        if not unthemed_events:
            logger.info("[APLR] 没有新的未分类事件。")
            return
            
        # 2. 获取现有主题（从向量库检索）
        theme_collection = self.vector_db.client.get_or_create_collection("themes")
        themes_data = theme_collection.get(include=['embeddings'])
        if not themes_data['ids']:
            logger.info("[APLR] 尚未执行过大固化，跳过增量更新。")
            return
            
        # 建立 theme_id 到向量的映射
        theme_id_to_centroid = {themes_data['ids'][i]: np.array(themes_data['embeddings'][i], dtype=np.float32) for i in range(len(themes_data['ids']))}
        
        threshold_str = self.db.get_consolidation_config("last_threshold")
        threshold = float(threshold_str) if threshold_str else 0.5
        
        # 获取全局高频词
        stopwords_str = self.db.get_consolidation_config("global_high_freq_words")
        global_stopwords = set(json.loads(stopwords_str)) if stopwords_str else set()
        
        event_ids = [e['event_id'] for e in unthemed_events]
        collection = self.vector_db.client.get_collection("events")
        res = collection.get(ids=event_ids, include=['embeddings'])
        id_to_embedding = {res['ids'][i]: res['embeddings'][i] for i in range(len(res['ids']))}
        
        # 建立 theme_id 到关键词的映射 (从 SQL 读取)
        sql_themes = {t['theme_id']: json.loads(t['keywords']) for t in self.db.get_all_thematic_memories()}

        affected_themes = set()
        for ev in unthemed_events:
            eid = ev['event_id']
            emb = np.array(id_to_embedding[eid], dtype=np.float32)
            emb = normalize(emb.reshape(1, -1))[0]
            
            # 使用向量库查询最接近的主题
            query_res = theme_collection.query(
                query_embeddings=[emb.tolist()],
                n_results=1
            )
            
            if query_res['ids'] and query_res['ids'][0]:
                best_theme_id = query_res['ids'][0][0]
                centroid = theme_id_to_centroid.get(best_theme_id)
                if centroid is None: continue
                
                dist = 1.0 - np.dot(emb, centroid)
                
                # 关键词加成
                theme_keys = set(sql_themes.get(best_theme_id, []))
                clean_text = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', ev['narrative']))
                ev_keys = set([w for w in jieba.lcut(clean_text) if len(w) > 1])
                
                # 剔除全局高频词，保持和大固化逻辑一致
                rare_ev_keys = ev_keys - global_stopwords
                
                common = theme_keys.intersection(rare_ev_keys)
                if common:
                    dist -= 0.05 * len(common)
                
                if dist < threshold:
                    logger.info(f"[APLR] 事件 {eid} 命中主题 {best_theme_id} (距离: {dist:.4f})")
                    self.db.map_event_to_theme(eid, best_theme_id)
                else:
                    logger.info(f"[APLR] 事件 {eid} 未命中任何主题 (最小距离: {dist:.4f})")

        # 3. 检查是否需要更新总结 (重心更新已移至此函数内部)
        await self.summarize_all_themes(force=False)

    async def summarize_all_themes(self, force=True):
        """为所有需要更新的主题生成总结"""
        themes = self.db.get_all_thematic_memories()
        for theme in themes:
            theme_id = theme['theme_id']
            event_count = theme['event_count']
            last_count = theme['last_summarized_count']
            
            # 如果是强制更新，或者新事件超过 5 条，或者新事件占比超过 20%
            if force or (event_count - last_count >= 5) or (last_count > 0 and (event_count - last_count) / last_count > 0.2):
                logger.info(f"[APLR] 正在为主题 {theme_id} 生成/更新总结...")
                events = self.db.get_events_by_theme(theme_id)
                summary = await self._generate_theme_summary(events, theme.get('summary', ""))
                if summary:
                    self.db.update_theme_summary(theme_id, summary, event_count)
                    
                    # 同步更新向量数据库中的重心
                    try:
                        event_collection = self.vector_db.client.get_collection("events")
                        t_event_ids = [te['event_id'] for te in events]
                        res = event_collection.get(ids=t_event_ids, include=['embeddings'])
                        if res.get('embeddings') is not None and len(res['embeddings']) > 0:
                            embs = np.array(res['embeddings'], dtype=np.float32)
                            new_centroid = np.mean(embs, axis=0)
                            
                            theme_collection = self.vector_db.client.get_or_create_collection("themes")
                            theme_collection.upsert(
                                ids=[theme_id],
                                embeddings=[new_centroid.tolist()],
                                metadatas=[{"type": "theme"}]
                            )
                            logger.info(f"[APLR] 主题 {theme_id} 的重心向量已随总结同步更新。")
                    except Exception as ve:
                        logger.error(f"[APLR] 更新主题 {theme_id} 向量失败: {ve}")

    async def _generate_theme_summary(self, events: List[dict], old_summary: str = "") -> str:
        """调用 LLM 生成主题总结"""
        formatted_events = []
        for e in events:
            ev_str = f"- [{e['date']}] {e['narrative']}"
            if e.get('emotion'):
                ev_str += f" (情绪: {e['emotion']})"
            formatted_events.append(ev_str)
            
        events_text = "\n".join(formatted_events)
        old_summary_context = f"\n你之前对这个主题的总结是：\n{old_summary}\n" if old_summary else ""
        
        system_prompt = self.persona
        prompt = f"""【当前任务】
{self.prompt_theme_summary}

以下是属于同一个主题的一组记忆事件：
{events_text}
{old_summary_context}

请以此为基础，写下你对这个主题的精炼总结。"""

        try:
            llm_resp = await self.llm_generate(prompt=prompt, system_prompt=system_prompt)
            return llm_resp.completion_text.strip()
        except Exception as e:
            logger.error(f"生成主题总结失败: {e}")
            return ""
