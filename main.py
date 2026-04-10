import os
import json
import math
import re
import jieba
import jieba.analyse
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, llm_tool
from astrbot.api.provider import ProviderRequest
from astrbot.core.conversation_mgr import Conversation
from .database import MemoryDB
from .summarizer import DailySummarizer
from .chat_history_extract import clean_dialogue_with_different_limits
import random
import shutil
import subprocess
import sys

@register("local_reminiscence", "olozhika", "基于定时总结和向量化的本地记忆插件", "1.2.2")
class LocalReminiscencePlugin(Star):
    def __init__(self, context: Context, config: any = None):
        super().__init__(context)
        self.context = context
        self.config = config if config else {}

        # 预先获取离线模式和镜像设置，确保后续操作（包括依赖检查）尊重这些设置
        def to_bool(value, default=False):
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            if isinstance(value, str):
                return value.strip().lower() in ["1", "true", "yes", "on"]
            return bool(value)

        self.offline_mode = to_bool(self.config.get("offline_mode", False))
        self.hf_endpoint = self.config.get("hf_endpoint", "https://hf-mirror.com").strip()

        if self.offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
            if self.hf_endpoint:
                os.environ["HF_ENDPOINT"] = self.hf_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)

        # 动态检测并安装 torch (CPU版)
        try:
            import torch
            import sentence_transformers
            import chromadb
            logger.info("[APLR] 已检测到 AI 相关依赖，跳过安装。")
        except ImportError:
            if self.offline_mode:
                logger.warning("[APLR] 离线模式已开启，但未检测到 AI 依赖。请先关闭离线模式以安装依赖。")
            else:
                logger.info("[APLR] 未检测到 AI 依赖，正在后台安装 CPU 版 torch 及相关库 (这可能需要几分钟)...")
                try:
                    # 1. 先安装 torch CPU 版
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "torch", 
                        "--index-url", "https://download.pytorch.org/whl/cpu"
                    ])
                    # 2. 再安装 sentence-transformers 和 chromadb
                    pip_cmd = [sys.executable, "-m", "pip", "install", "sentence-transformers", "chromadb"]
                    if self.hf_endpoint:
                        # 如果配置了 hf_endpoint (通常是境内镜像)，则使用国内 PyPI 镜像加速依赖安装
                        pip_cmd.extend(["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
                    subprocess.check_call(pip_cmd)
                    logger.info("[APLR] AI 依赖安装成功！")
                except Exception as e:
                    logger.error(f"[APLR] AI 依赖安装失败: {e}")
        
        # 在确保 torch 可能已安装后，再导入依赖 torch 的模块
        from .vector_db import VectorDB
        
        # 获取配置项
        dialog_folder_rel = self.config.get("dialog_folder", "APLR_chat_history")
        memory_db_path_rel = self.config.get("memory_db_path", "APLR_DailyReview.db")
        vector_db_path_rel = self.config.get("vector_db_path", "APLR_VectorDB")

        data_dir = StarTools.get_data_dir()
        
        # 路径解析逻辑：优先考虑绝对路径，否则相对于插件数据目录
        def resolve_path(path_str, data_path):
            p = Path(path_str)
            if p.is_absolute():
                return p
            return (data_path / p).resolve()

        self.dialog_folder = resolve_path(dialog_folder_rel, data_dir)
        self.db_path = resolve_path(memory_db_path_rel, data_dir)
        self.vector_db_path = resolve_path(vector_db_path_rel, data_dir)
        self.target_user_id_list = self.config.get("target_user_id_list", ["Lanya_QQ:FriendMessage:947628188"])
        self.username = self.config.get("username", "olozhika")
        self.ai_name = self.config.get("ai_name", "Lanya")
        embedding_model = self.config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
        embedding_cache_dir_rel = self.config.get("embedding_cache_dir", "APLR_ModelCache")

        embedding_trust_remote_code = to_bool(self.config.get("embedding_trust_remote_code", False))
        embedding_cache_dir = None
        if embedding_cache_dir_rel:
            embedding_cache_dir = resolve_path(embedding_cache_dir_rel, data_dir)

        # 确保目录存在
        self.dialog_folder.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        if embedding_cache_dir:
            embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self.db = MemoryDB(str(self.db_path))
        self.vector_db = VectorDB(
            str(self.vector_db_path),
            model_name=embedding_model,
            model_cache_dir=str(embedding_cache_dir) if embedding_cache_dir else None,
            hf_endpoint=self.hf_endpoint,
            trust_remote_code=embedding_trust_remote_code,
            offline_mode=self.offline_mode,
            ai_name=self.ai_name
        )

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在生成回复前注入历史记忆，帮助 AI 维持长期记忆"""
        try:
            message_str = event.message_str
            logger.debug(f"[APLR] 收到消息内容 (前100字): {message_str[:100]}...")
            
            # 1. 检测是否为新对话开启
            uid = event.unified_msg_origin
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(uid)
            
            is_new_session = False
            if curr_cid is None:
                is_new_session = True
            else:
                conversation: Conversation = await conv_mgr.get_conversation(uid, curr_cid)
                # 下面这行这么写纯粹是因为作者olozhika不熟悉json格式...
                if not conversation or not conversation.history or conversation.history == '[]' or json.loads(conversation.history) == []:
                    is_new_session = True
            
            # 如果是新对话开启，则注入历史摘要
            if is_new_session:
                # 2. 获取最近3天的总结
                summaries = self.db.get_summaries(3)
                # 3. 获取最近7天的感悟
                reflections = self.db.get_reflections(7)
                
                if summaries or reflections:
                    memory_text = "\n\n【历史记忆 - 这是你对过去几天的回忆】\n"
                    
                    if summaries:
                        memory_text += "### 最近3天的深度回忆\n"
                        now = datetime.now()
                        for s in summaries:
                            # 计算日期差距
                            try:
                                event_date = datetime.strptime(s['date'], "%Y-%m-%d")
                                days_diff = (now.date() - event_date.date()).days
                            except Exception:
                                days_diff = 999 
                            
                            # 确定阈值
                            if days_diff <= 1: # 今天和昨天 (2天内)
                                threshold = 24
                            elif days_diff <= 2: # 前天 (3天内)
                                threshold = 30
                            else:
                                threshold = 999 

                            memory_text += f"📅 日期: {s['date']}\n"
                            memory_text += f"💭 当时的感悟: {s['reflection']}\n"
                            
                            # 过滤事件
                            filtered_events = []
                            if s['events']:
                                for e in s['events']:
                                    importance = e.get('importance', 0)
                                    intensity = e.get('emotional_intensity', 0)
                                    if importance * intensity >= threshold:
                                        filtered_events.append(e)
                            
                            if filtered_events:
                                memory_text += "📍 发生的重要事件:\n"
                                for e in filtered_events:
                                    tags_str = f" #{' #'.join(e['tags'])}" if e['tags'] else ""
                                    memory_text += f"  - {e['narrative']} ({e['emotion']}) [ID: {e['event_id']}]\n"
                                if self.config.get("encourage_deep_recall", False):
                                    memory_text += "注：若想进一步回忆某个事件的细节或感想，可以使用 `recall_event_reflection_tool` 并传入对应的 事件ID来深度回想；如果觉得回忆还不够充分，可以使用 `recall_memory_tool` 对刚想起的片段进行联想）\n"
                            memory_text += "---\n"
                    
                    if reflections:
                        summary_dates = {s['date'] for s in summaries}
                        other_reflections = [r for r in reflections if r['date'] not in summary_dates]
                        
                        if other_reflections:
                            memory_text += "### 过去7天的感悟片段\n"
                            for r in other_reflections:
                                memory_text += f"📅 {r['date']} | 感悟: {r['reflection']}\n"
                    
                    # 注入到 system_prompt
                    req.system_prompt += memory_text
                    logger.info(f"[APLR] 检测到新对话开启，已为会话注入历史记忆，长度: {len(memory_text)}")

            # 0. 提取当前用户的 Nickname (增强版逻辑)
            current_nickname = None
            
            # 策略 A: 尝试从标签中提取
            tag_match = re.search(r'<system_reminder>(.*?)</system_reminder>', message_str, re.IGNORECASE | re.DOTALL)
            if tag_match:
                tag_content = tag_match.group(1)
                nick_match = re.search(r'Nickname:\s*([^,\s]+)', tag_content, re.IGNORECASE)
                if nick_match:
                    current_nickname = nick_match.group(1).strip()
                    logger.debug(f"[APLR] 从标签中成功提取 Nickname: {current_nickname}")
            
            # 策略 B: 兜底方案，如果标签提取失败，直接全文搜索关键词
            if not current_nickname:
                nick_match = re.search(r'Nickname:\s*([^,\s]+)', message_str, re.IGNORECASE)
                if nick_match:
                    current_nickname = nick_match.group(1).strip()
                    logger.debug(f"[APLR] 标签提取失败，通过全文搜索提取 Nickname: {current_nickname}")
            
            if not current_nickname:
                logger.debug(f"[APLR] 无法从消息中提取 Nickname")
            
            # 清理消息内容（去除系统提示词部分，避免干扰检索）
            clean_message = re.sub(r'<system_reminder>.*?</system_reminder>', '', message_str, flags=re.DOTALL).strip()
            if not clean_message:
                clean_message = message_str
            
            logger.debug(f"[APLR] 清理后的消息内容: {clean_message}")
            
            # 1. 获取所有相关节点
            all_nodes = []
            seen_names = set()
            
            # 首先尝试获取用户本人节点
            if current_nickname:
                logger.debug(f"[APLR] 正在搜索用户节点: {current_nickname}")
                user_nodes = self.db.search_nodes(current_nickname, limit=1, include_description=False)
                if user_nodes:
                    node = user_nodes[0]
                    logger.debug(f"[APLR] 找到用户节点: {node['name']} (ID: {node['id']})")
                    node['is_user'] = True # 标记为当前聊天对象
                    all_nodes.append(node)
                    seen_names.add(node['name'])
                else:
                    logger.debug(f"[APLR] 未找到用户节点: {current_nickname}")

            # 获取消息相关的节点
            msg_nodes, _ = await self._get_nodes_context(message_str, include_description=False)
            for node in msg_nodes:
                if node['name'] not in seen_names:
                    all_nodes.append(node)
                    seen_names.add(node['name'])

            if all_nodes:
                context_parts = []
                for n in all_nodes:
                    last_updated_date = n['last_updated'].split(' ')[0] if n['last_updated'] else "未知"
                    context_parts.append(f"📌 {n['name']}，{n['description']}。(最后更新: {last_updated_date})")
                
                injection_text = f"\n\n【记忆节点背景 - 这是你对提及实体及当前聊天对象的已知认知】\n" + "\n\n".join(context_parts) + "\n"
                req.system_prompt += injection_text
                logger.info(f"[APLR] 自动注入了 {len(all_nodes)} 个记忆节点: {', '.join([n['name'] for n in all_nodes])}")

            # 0. 自动触发记忆检索 (语义匹配) - 放在新会话注入后面，且非新会话也会触发
            keywords = ["之前", "记得", "回忆", "想起", "以前", "过去"]
            
            auto_recall_prob = self.config.get("auto_recall_probability", 0.3)
            auto_recall_threshold = self.config.get("auto_recall_threshold", 40.0)
            
            # 命中关键词或概率触发
            hit_keyword = any(kw in clean_message for kw in keywords)
            hit_prob = random.random() <= auto_recall_prob
            
            if hit_keyword or hit_prob:
                # 构造检索词：用户名: 用户发言
                if current_nickname:
                    search_query = f"{current_nickname}: {clean_message}"
                else:
                    search_query = clean_message
                
                logger.debug(f"[APLR] 触发记忆检索，检索词: {search_query}")
                
                top_n = self.config.get("top_n_events", 10)
                m1 = self.config.get("m1_top_events", 3)
                m2 = self.config.get("m2_random_events", 2)
                
                search_results = self.vector_db.search_events(search_query, top_n=top_n)
                if search_results:
                    full_events = []
                    for res in search_results:
                        if res['relevance'] >= auto_recall_threshold:
                            ev = self.db.get_event_by_id(res['event_id'])
                            if ev:
                                ev['relevance'] = res['relevance']
                                full_events.append(ev)
                    
                    if full_events:
                        ranked_events = self._rank_events(full_events, search_query)
                        selected_events = ranked_events[:m1]
                        remaining = ranked_events[m1:]
                        if remaining and m2 > 0:
                            selected_events.extend(random.sample(remaining, min(len(remaining), m2)))
                        
                        if selected_events:
                            # 强化记忆
                            re_intensity = self.config.get("reinforcement_intensity", 1.0)
                            if re_intensity > 0:
                                event_ids = [ev['event_id'] for ev in selected_events]
                                self.db.reinforce_memory(event_ids)

                            # 使用聚类和叙述桥梁
                            clusters = self._cluster_events_by_context(selected_events)
                            recall_text = "\n\n【自动回想 - 你回想起了以下记忆片段】\n"
                            recall_text += self._generate_narrative_bridge(clusters)
                            
                            if self.config.get("encourage_deep_recall", False):
                                recall_text += "（注：若想进一步回忆某个事件的细节或感想，可以使用 `recall_event_reflection_tool` 并传入对应的 事件ID来深度回想；如果觉得回忆还不够充分，可以使用 `recall_memory_tool` 对刚想起的片段进行联想）"
                            req.system_prompt += recall_text
                            trigger_reason = "关键词" if hit_keyword else "概率"
                            logger.info(f"[APLR] 自动触发记忆检索({trigger_reason})，注入了 {len(selected_events)} 条相关记忆")
            
        except Exception as e:
            logger.error(f"[APLR] 注入历史记忆失败: {e}", exc_info=True)

    async def _get_nodes_context(self, text: str, include_description: bool = False, max_nodes: int = 8) -> tuple[list[dict], list[str]]:
        """根据文本提取关键词并获取相关记忆节点对象"""
        try:
            # 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=15)
            if not keywords:
                return [], []
            
            # 搜索节点
            nodes = []
            seen_names = set()
            
            for kw in keywords:
                if len(nodes) >= max_nodes:
                    break
                    
                is_common_name = (kw.lower() in [self.username.lower(), self.ai_name.lower()])
                
                if is_common_name:
                    found_nodes = self.db.get_nodes_by_names([kw])
                else:
                    found_nodes = self.db.search_nodes(kw, include_description=include_description)
                
                for node in found_nodes:
                    if node['name'] not in seen_names:
                        nodes.append(node)
                        seen_names.add(node['name'])
                        if len(nodes) >= max_nodes:
                            break
            
            node_names = [n['name'] for n in nodes]
            return nodes, node_names
        except Exception as e:
            logger.error(f"获取节点背景失败: {e}")
            return [], []

    @filter.command("daily_summary_command")
    async def daily_summary_command(self, event: AstrMessageEvent):
        """进行每日总结用的工具，回顾并总结今日或指定日期的交流。用法：/daily_summary_command [日期]（可选，默认今天，格式YYYY-MM-DD）。人类专用版！"""
        async for result in self._daily_summary_logic(event):
            yield result

    @filter.command("extract_chat_history_command")
    async def extract_chat_history_command(self, event: AstrMessageEvent):
        """手动提取指定日期的聊天记录。用法：/extract_chat_history_command [日期]（格式YYYY-MM-DD）。"""
        args = event.message_str.strip().split()
        if len(args) < 2:
            yield event.plain_result("请提供日期，格式为 YYYY-MM-DD")
            return
        
        date_str = args[1]
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            yield event.plain_result("日期格式不正确，请使用 YYYY-MM-DD")
            return
        
        yield event.plain_result(f"正在提取 {date_str} 的聊天记录...")
        
        try:
            core_db_path = Path.cwd() / "data" / "data_v4.db"
            for target_user_id in self.target_user_id_list:
                clean_dialogue_with_different_limits(
                    db_path=core_db_path,
                    output_dir=self.dialog_folder,
                    username=self.username,
                    ai_name=self.ai_name,
                    platform="AstrBot",
                    target_date=date_str,
                    target_user_id=target_user_id
                )
            yield event.plain_result(f"✅ 已成功提取 {date_str} 的聊天记录。")
        except Exception as e:
            logger.error(f"提取聊天记录失败: {e}")
            yield event.plain_result(f"❌ 提取聊天记录失败: {e}")

    @filter.command("vectorize_events")
    async def vectorize_events_command(self, event: AstrMessageEvent):
        """将指定日期的事件向量化并存入向量数据库。用法：/vectorize_events [日期|all]（格式YYYY-MM-DD）。"""
        args = event.message_str.strip().split()
        if len(args) < 2:
            yield event.plain_result("请提供日期（YYYY-MM-DD）或输入 'all' 重新向量化全部。")
            return
        
        param = args[1].lower()
        
        if param == "all":
            yield event.plain_result("⚠️ 正在清空向量库并重新录入所有事件，这可能需要较长时间，请稍候...")
            try:
                # 1. 清空向量库
                self.vector_db.clear_all()
                
                # 2. 获取所有事件
                all_events = self.db.get_all_events()
                if not all_events:
                    yield event.plain_result("数据库中没有找到任何事件。")
                    return
                
                # 3. 分批向量化（防止内存溢出或超时）
                batch_size = 50
                total = len(all_events)
                for i in range(0, total, batch_size):
                    batch = all_events[i:i + batch_size]
                    self.vector_db.add_events(batch)
                    if i % 100 == 0 or i + batch_size >= total:
                        logger.info(f"[APLR] 向量化进度: {min(i + batch_size, total)}/{total}")
                
                yield event.plain_result(f"✅ 已成功重新向量化全部 {total} 个事件。")
            except Exception as e:
                logger.error(f"重新向量化全部失败: {e}")
                yield event.plain_result(f"❌ 重新向量化全部失败: {e}")
            return

        # 原有的按日期向量化逻辑
        date_str = param
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            yield event.plain_result("日期格式不正确，请使用 YYYY-MM-DD 或输入 'all'")
            return
        
        events = self.db.get_events_by_date(date_str)
        if not events:
            yield event.plain_result(f"没找到 {date_str} 的事件记录。")
            return
        
        try:
            self.vector_db.add_events(events)
            yield event.plain_result(f"✅ 已成功将 {date_str} 的 {len(events)} 个事件向量化。")
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            yield event.plain_result(f"❌ 向量化失败: {e}")

    @filter.command("update_nodes_command")
    async def update_nodes_command(self, event: AstrMessageEvent):
        """根据指定日期的已有事件提取并更新记忆节点。用法：/update_nodes_command [日期]（格式YYYY-MM-DD）。"""
        args = event.message_str.strip().split()
        if len(args) < 2:
            yield event.plain_result("请提供日期，格式为 YYYY-MM-DD")
            return
            
        date_str = args[1]
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            yield event.plain_result("日期格式不正确，请使用 YYYY-MM-DD")
            return
            
        events = self.db.get_events_by_date(date_str)
        if not events:
            yield event.plain_result(f"没找到 {date_str} 的事件记录，无法提取节点。")
            return
            
        yield event.plain_result(f"正在从 {date_str} 的 {len(events)} 个事件中提取记忆节点...")
        
        try:
            # 初始化总结器 (需要 LLM Provider)
            umo = event.unified_msg_origin
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            if not provider_id:
                yield event.plain_result("❌ 暂时连接不到大脑（LLM Provider），请检查配置。")
                return

            async def llm_generate_func(prompt, system_prompt):
                return await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=[]
                )

            # 3. 获取基础提示词
            base_system_prompt = self.config.get("default_prompt", "") if self.config else ""
            base_user_prompt = self.config.get("user_prompts", "") if self.config else ""

            # 注入 Astrbot 自带的人格提示词
            try:
                persona = await self.context.persona_manager.get_default_persona_v3(umo=umo)
                if persona and persona.get('prompt'):
                    base_system_prompt = (persona['prompt'] + "\n" + base_system_prompt) if base_system_prompt else (persona['prompt'] + "\n")
                    logger.debug(f"已注入Astrbot人格设定")
                    logger.debug(f"base sys prompt: {base_system_prompt}")
            except Exception as pe:
                logger.warning(f"[APLR] 尝试获取 Astrbot 人格设定失败: {pe}")

            summarizer = DailySummarizer(
                llm_generate_func=llm_generate_func,
                ai_name=self.ai_name,
                base_system_prompt=base_system_prompt,
                base_user_prompt=base_user_prompt,
                search_events_func=self._search_events_for_summary
            )

            # 获取现有节点背景 (提取节点时也放宽上限，允许描述匹配)
            nodes_objs, _ = await self._get_nodes_context("\n".join([e['narrative'] for e in events]), include_description=True, max_nodes=100)
            existing_nodes_context = ""
            if nodes_objs:
                existing_nodes_context = "\n".join([f"- {n['name']}: {n['description']}" for n in nodes_objs])

            nodes_result = await summarizer.extract_nodes_from_events(events, date_str, existing_nodes_context=existing_nodes_context)
            if nodes_result:
                nodes, deleted_nodes = nodes_result
                if nodes:
                    self.db.update_nodes(nodes)
                if deleted_nodes:
                    self.db.delete_nodes(deleted_nodes)
                
                node_names = [n.name for n in nodes]
                del_msg = f"，并删除了 {len(deleted_nodes)} 个冗余节点" if deleted_nodes else ""
                yield event.plain_result(f"✅ 已成功更新 {len(nodes)} 个记忆节点：{', '.join(node_names)}{del_msg}")
            else:
                yield event.plain_result(f"⚠️ 未能从 {date_str} 的事件中提取到新的记忆节点。")
        except Exception as e:
            logger.error(f"提取节点失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 提取节点失败: {e}")

    @filter.command("recall_memory_command")
    async def recall_memory_command(self, event: AstrMessageEvent):
        """根据输入文本检索最相关的记忆。用法：/recall_memory_command [文本内容] [条数(可选)]。人类专用版！"""
        msg = event.message_str.strip()
        # 移除可能的斜杠和命令名
        msg = re.sub(r'^/?recall_memory_command\s*', '', msg).strip()
        
        if not msg:
            yield event.plain_result("请输入要检索的内容。")
            return
        
        # 尝试解析末尾的数字作为条数
        parts = msg.split()
        count = None
        query = msg
        if len(parts) > 1 and parts[-1].isdigit():
            count = int(parts[-1])
            query = " ".join(parts[:-1])
        
        resp = await self._get_memory_retrieval_text(query, count=count)
        yield event.plain_result(resp)

    @llm_tool(name="recall_memory_tool")
    async def recall_memory_tool(self, event: AstrMessageEvent, query: str = "", count: int = 0) -> str:
        """根据输入文本从你的长期记忆中检索最相关的事件。当你需要回忆过去发生的事情时使用。
        
        Args:
            query(string): 文本描述。
            count(int): 返回条数（可选，默认为系统默认值）。
        """
        if not query:
            return "错误：需要提供检索关键词query"
        
        # 1. 检索
        result = await self._get_memory_retrieval_text(query, count=count if count > 0 else None)
        
        # 2. 强化被回忆的事件
        re_intensity = self.config.get("reinforcement_intensity", 1.0)
        if re_intensity > 0 and result and "暂时没有找到" not in result and "未找到" not in result:
            # 提取 ID
            event_ids = re.findall(r'\[ID: (evt_\d{8}_\d{3})\]', result)
            if event_ids:
                self.db.reinforce_memory(event_ids)
                logger.info(f"[APLR] 强化了 {len(event_ids)} 条记忆的权重")
                
        return result

    @llm_tool(name="recall_event_reflection_tool")
    async def recall_event_reflection_tool(self, event: AstrMessageEvent, event_id: str) -> str:
        """获取特定事件的深度观察和感想。当你看到某个事件的简述（narrative）并希望回想更多细节或当时的心理活动时使用。
        
        Args:
            event_id(string): 事件的唯一ID（如 evt_20260312_001）。
        """
        if not event_id:
            return "错误：需要提供事件ID event_id。"
        return await self._get_event_reflection_logic(event_id)

    async def _get_event_reflection_logic(self, event_id: str) -> str:
        ev = self.db.get_event_by_id(event_id)
        if ev:
            reflection = ev.get('reflection', '')
            if reflection:
                res = f"关于事件 {event_id} 的深度观察和感想：\n{reflection}"
            else:
                res = f"该事件 ({event_id}) 暂时没有记录深度感想呢。"
            
            if self.config.get("encourage_deep_recall", False):
                res += f"\n（注：若希望回忆该事件发生当天 ({ev['date']}) 的整体心境，可选择调用 `recall_daily_reflection_tool`。）"
            return res
        return f"未找到 ID 为 {event_id} 的事件。"

    @llm_tool(name="recall_daily_reflection_tool")
    async def recall_daily_reflection_tool(self, event: AstrMessageEvent, date: str) -> str:
        """获取特定日期的每日自由心得（感悟）。想回忆某天的整体心境时使用。
        
        Args:
            date(string): 日期，格式为 YYYY-MM-DD。
        """
        if not date:
            return "错误：需要提供日期 date。"
        return await self._get_daily_reflection_logic(date)

    @llm_tool(name="recall_recent_events_tool")
    async def recall_recent_events_tool(self, event: AstrMessageEvent, days: int = 7, min_score: int = 20) -> str:
        """获取最近一段时间内比较重要的、或情感强烈的事件。用于快速回顾近期的重点记忆。
        
        Args:
            days(int): 回溯的天数，默认为 7 天。
            min_score(int): 筛选阈值（重要性 * 情感强度），默认为 20。分数越高，筛选出的事件越少且越重要。
        """
        try:
            events = self.db.get_recent_important_events(days, min_score)
            if not events:
                return f"在最近 {days} 天内，没有找到重要性评分（重要性 * 情感强度）大于等于 {min_score} 的事件。"
            
            # 聚类并格式化
            clusters = self._cluster_events_by_context(events)
            resp = f"📅 检索到了最近 {days} 天内的重点记忆（评分 >= {min_score}）：\n\n"
            resp += self._generate_narrative_bridge(clusters)
            return resp
        except Exception as e:
            logger.error(f"执行 recall_recent_events_tool 失败: {e}", exc_info=True)
            return f"执行失败: {e}"

    async def _get_daily_reflection_logic(self, date: str) -> str:
        ref = self.db.get_reflection_by_date(date)
        if ref:
            reflection = ref.get('reflection', '')
            if reflection:
                return f"关于 {date} 的整体感悟是：\n{reflection}"
            else:
                return f"那天 ({date}) 好像没写下什么特别的感悟呢。"
        return f"数据库里没找到关于 {date} 的感悟记录。"

    @filter.command("recall_event_reflection_command")
    async def recall_event_reflection_command(self, event: AstrMessageEvent, event_id: str = ""):
        """获取特定事件的深度观察和感想。用法：/recall_event_reflection_command [事件ID]"""
        event_id = event_id.strip()
        if not event_id:
            yield event.plain_result("请输入事件ID（如 evt_20260312_001）。")
            return
        
        res = await self._get_event_reflection_logic(event_id)
        yield event.plain_result(res)

    @filter.command("recall_daily_reflection_command")
    async def recall_daily_reflection_command(self, event: AstrMessageEvent, date: str = ""):
        """获取特定日期的每日自由心得（感悟）。用法：/recall_daily_reflection_command [日期(YYYY-MM-DD)]"""
        date = date.strip()
        if not date:
            yield event.plain_result("请输入日期（如 2026-03-12）。")
            return
        
        res = await self._get_daily_reflection_logic(date)
        yield event.plain_result(res)

    @filter.command("rebuild_relations_command")
    async def rebuild_relations_command(self, event: AstrMessageEvent, date: str = ""):
        """为特定日期的事件重新发现跨日关联。用法：/rebuild_relations_command [日期(YYYY-MM-DD)]"""
        date = date.strip()
        if not date:
            yield event.plain_result("请输入日期（如 2026-03-12）。")
            return
        
        # 1. 获取该日期的所有事件
        events = self.db.get_events_by_date(date)
        if not events:
            yield event.plain_result(f"数据库中未找到日期为 {date} 的事件。")
            return
        
        yield event.plain_result(f"正在为 {date} 的 {len(events)} 个事件重新发现跨日关联，请稍候...")
        
        # 2. 初始化总结器
        provider_id = self.config.get("llm_provider_id")
        if not provider_id:
            yield event.plain_result("未配置 LLM Provider，无法执行关联发现。")
            return
            
        async def llm_generate_func(prompt, system_prompt):
            return await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=system_prompt,
                history=[]
            )
            
        # 获取基础提示词并注入 Astrbot 自带的人格提示词
        base_system_prompt = self.config.get("default_prompt", "") if self.config else ""
        base_user_prompt = self.config.get("user_prompts", "") if self.config else ""
        try:
            umo = event.unified_msg_origin
            persona = await self.context.persona_manager.get_default_persona_v3(umo=umo)
            if persona and persona.get('prompt'):
                base_system_prompt = (persona['prompt'] + "\n" + base_system_prompt) if base_system_prompt else (persona['prompt'] + "\n")
                logger.debug(f"已注入Astrbot人格设定")
                logger.debug(f"base sys prompt: {base_system_prompt}")
        except Exception as pe:
            logger.warning(f"[APLR] 尝试获取 Astrbot 人格设定失败: {pe}")

        summarizer = DailySummarizer(
            llm_generate_func=llm_generate_func,
            ai_name=self.ai_name,
            base_system_prompt=base_system_prompt,
            base_user_prompt=base_user_prompt,
            search_events_func=self._search_events_for_summary
        )
        
        # 3. 发现关联
        try:
            relations = await summarizer.discover_cross_day_relations(events, date)
            if relations:
                self.db.insert_relations(relations)
                yield event.plain_result(f"成功为 {date} 发现了 {len(relations)} 条新的跨日关联并已存入数据库。")
            else:
                yield event.plain_result(f"未发现 {date} 的事件与往日记忆有实质性关联。")
        except Exception as e:
            logger.error(f"执行 rebuild_relations_command 失败: {e}", exc_info=True)
            yield event.plain_result(f"执行失败: {e}")

    @filter.command("recall_recent_events_command")
    async def recall_recent_events_command(self, event: AstrMessageEvent, days: str = "7", min_score: str = "20"):
        """获取最近一段时间内比较重要的、或情感强烈的事件。用法：/recall_recent_events_command [天数] [筛选分数]"""
        try:
            d = int(days)
            s = int(min_score)
        except ValueError:
            yield event.plain_result("请输入有效的数字（如：/recall_recent_events_command 7 20）。")
            return
            
        res = await self.recall_recent_events_tool(event, d, s)
        yield event.plain_result(res)

    @filter.command("recall_node_command")
    async def recall_node_command(self, event: AstrMessageEvent):
        """搜索某个记忆节点（实体或概念）。用法：/recall_node_command [名称]"""
        msg = event.message_str.strip()
        # 移除可能的斜杠和命令名
        query = re.sub(r'^/?recall_node_command\s*', '', msg).strip()
        
        if not query:
            yield event.plain_result("请输入要搜索的节点名称。")
            return
        
        nodes = self.db.search_nodes(query)
        if nodes:
            resp = f"💡 找到了 {len(nodes)} 个相关的记忆节点：\n\n"
            for node in nodes:
                resp += f"📌 **{node['name']}** ({node['type']})\n"
                resp += f"📝 描述: {node['description']}\n"
                last_updated_date = node['last_updated'].split(' ')[0] if node['last_updated'] else "未知"
                resp += f"🕒 最后更新: {last_updated_date}\n"
                resp += "\n"
            yield event.plain_result(resp)
        else:
            yield event.plain_result(f"暂时没有关于“{query}”的节点记忆呢。")

    @filter.command("write_node_command")
    async def write_node_command(self, event: AstrMessageEvent):
        """手动写入或更新记忆节点。用法：/write_node_command [节点名] [类型] [描述]"""
        msg = event.message_str.strip()
        # 移除命令名
        content = re.sub(r'^/?write_node_command\s*', '', msg).strip()
        
        if not content:
            yield event.plain_result("用法：/write_node_command [节点名] [类型] [描述]")
            return
            
        parts = content.split(maxsplit=2)
        if len(parts) < 3:
            yield event.plain_result("参数不足。用法：/write_node_command [节点名] [类型] [描述]\n示例：/write_node_command olozhika 人物 Lanya的好朋友")
            return
            
        name, node_type, description = parts
        
        from .models import MemoryNode
        node = MemoryNode(name=name, type=node_type, description=description)
        
        try:
            self.db.update_nodes([node])
            yield event.plain_result(f"✅ 已成功写入/更新节点：{name} ({node_type})")
        except Exception as e:
            logger.error(f"写入节点失败: {e}")
            yield event.plain_result(f"❌ 写入节点失败: {e}")

    @llm_tool(name="write_node_tool")
    async def write_node_tool(self, event: AstrMessageEvent, name: str, node_type: str, description: str) -> str:
        """手动写入或更新记忆节点（如人物、地点、核心概念）。当你有重要信息需要迫切记录、更新时使用。不急的内容可以等到每日总结环节自动更新。
        
        注意：
        1. 请务必先使用 `recall_node_tool` 查询相应节点是否存在。
        2. 如果节点已存在，请在保留完整核心事实（如身份、关键背景）的基础上，根据今日信息更新其状态或追加新进展。
        
        Args:
            name(string): 节点名称。
            node_type(string): 节点类型（如：人物、地点、物品、概念、组织）。
            description(string): 节点的详细描述或背景信息。
        """
        if not name or not node_type or not description:
            return "错误：name, node_type 和 description 均为必填项。"
            
        from .models import MemoryNode
        node = MemoryNode(name=name, type=node_type, description=description)
        
        try:
            self.db.update_nodes([node])
            return f"✅ 已成功写入/更新节点：{name} ({node_type})"
        except Exception as e:
            logger.error(f"写入节点失败: {e}")
            return f"❌ 写入节点失败: {e}"

    @llm_tool(name="recall_node_tool")
    async def recall_node_tool(self, event: AstrMessageEvent, name: str) -> str:
        """搜索某个特定的记忆节点（如人物、地点、核心概念）。当你需要了解某个特定对象或概念的背景信息时使用。
        
        Args:
            name(string): 要搜索的节点名称。
        """
        nodes = self.db.search_nodes(name)
        if nodes:
            res = f"找到以下关于 {name} 的相关 node 信息：\n"
            for node in nodes:
                last_updated_date = node['last_updated'].split(' ')[0] if node['last_updated'] else "未知"
                res += f"- {node['name']} ({node['type']}): {node['description']} (最后更新于 {last_updated_date})\n"
            return res
        return f"未找到关于 {name} 的节点信息。"

    async def _get_memory_retrieval_text(self, query: str, count: int = None) -> str:
        # 1. 获取配置
        if count:
            m1 = count
            m2 = round(count / 2)
            top_n = round(count * 2.5)
        else:
            top_n = self.config.get("top_n_events", 10)
            m1 = self.config.get("m1_top_events", 3)
            m2 = self.config.get("m2_random_events", 2)

        # 2. 向量检索
        search_query = query
        if self.ai_name:
            # 使用正则替换 "我"，但排除 "我们" 和 "自我"
            pattern = r'(?<!自)我(?![们])'
            search_query = re.sub(pattern, self.ai_name, query)
            
        search_results = self.vector_db.search_events(search_query, top_n=top_n)
        if not search_results:
            return "暂时没有找到相关的记忆呢。"

        # 3. 获取完整信息并合并相关度
        full_events = []
        for res in search_results:
            eid = res['event_id']
            relevance = res['relevance']
            ev = self.db.get_event_by_id(eid)
            if ev:
                ev['relevance'] = relevance 
                full_events.append(ev)
        
        if not full_events:
            return "检索到了 ID 但没能从数据库找到详细信息。"

        # 4. 排序
        ranked_events = self._rank_events(full_events, search_query)

        # 5. 选择 M1 + M2
        selected_events = ranked_events[:m1]
        remaining = ranked_events[m1:]
        if remaining and m2 > 0:
            selected_events.extend(random.sample(remaining, min(len(remaining), m2)))

        if selected_events:
            # 强化记忆
            re_intensity = self.config.get("reinforcement_intensity", 1.0)
            if re_intensity > 0:
                event_ids = [ev['event_id'] for ev in selected_events]
                self.db.reinforce_memory(event_ids)

        # 6. 聚类
        clusters = self._cluster_events_by_context(selected_events)

        # 7. 格式化输出
        resp = f"🔍 回想起了相关记忆：\n\n"
        
        # 提取相关背景知识
        node_names = set()
        for ev in selected_events:
            with self.db._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.name FROM tags t
                    JOIN event_tags et ON t.id = et.tag_id
                    WHERE et.event_id = ?
                """, (ev['event_id'],))
                tags = [row['name'] for row in cursor.fetchall()]
                node_names.update(tags)
        
        nodes = self.db.get_nodes_by_names(list(node_names))
        if nodes:
            resp += "💡 相关背景知识：\n"
            for node in nodes:
                last_updated_date = node['last_updated'].split(' ')[0] if node['last_updated'] else "未知"
                desc = node['description']
                resp += f"  - {node['name']} ({node['type']}) [最后更新: {last_updated_date}]: {desc}\n"
            resp += "\n"

        # 使用叙述桥梁
        resp += self._generate_narrative_bridge(clusters)
        
        if self.config.get("encourage_deep_recall", False):
            resp += "\n（注：若想进一步回忆某个事件的细节 or 感想，可以使用 `recall_event_reflection_tool` 并传入对应的 事件ID来深度回想；如果觉得回忆还不够充分，可以使用 `recall_memory_tool` 对刚想起的片段进行联想）"
        return resp

    def _generate_narrative_bridge(self, clusters: List[List[dict]]) -> str:
        """为聚类后的记忆片段生成叙述性桥梁"""
        if not clusters:
            return ""
        
        narrative = ""
        for i, cluster in enumerate(clusters):
            if len(clusters) > 1:
                narrative += f"【记忆片段 {i+1}】\n"
            
            dates = sorted(list(set(ev['date'] for ev in cluster)))
            if len(dates) > 1:
                narrative += f"（这段记忆跨越了 {dates[0]} 到 {dates[-1]}）\n"
            else:
                narrative += f"（发生在 {dates[0]}）\n"
            
            for ev in cluster:
                narrative += f"  - {ev['narrative']} [ID: {ev['event_id']}]\n"
            narrative += "---\n"
        return narrative

    def _rank_events(self, events: List[dict], query: str = "") -> List[dict]:
        """按照距今日期差、事件重要性、情感强度、关键词匹配度、连贯性以及强化计数进行综合排序"""
        now = datetime.now().date()
        
        # 预处理：计算局部关键词权重 (Simple Local IDF)
        word_weights = {}
        if query:
            query_terms = set([word for word in jieba.lcut(query) if len(word) > 1])
            
            if query_terms:
                doc_counts = {}
                for ev in events:
                    narrative = ev.get('narrative', '')
                    for term in query_terms:
                        if term in narrative:
                            doc_counts[term] = doc_counts.get(term, 0) + 1
                
                num_docs = len(events)
                for term in query_terms:
                    count = doc_counts.get(term, 0)
                    word_weights[term] = math.log(num_docs / (count + 1.0)) + 1.0

        w_time = self.config.get("weight_time", 1.0)
        w_importance = self.config.get("weight_importance", 1.0)
        w_intensity = self.config.get("weight_emotional_intensity", 1.0)
        w_rare = self.config.get("weight_rare_word", 1.0)

        def score_func(ev):
            try:
                ev_date = datetime.strptime(ev['date'], "%Y-%m-%d").date()
                days_diff = (now - ev_date).days
                date_weight = 1.0 / (1 + math.log1p(days_diff / 30.0))
            except:
                date_weight = 0.1
            
            importance = ev.get('importance', 1)
            intensity = ev.get('emotional_intensity', 1)
            relevance = (ev.get('relevance', 100) / 100.0) ** 4
            
            # 关键词加成
            keyword_score = 0.0
            narrative = ev.get('narrative', '')
            for term, weight in word_weights.items():
                if term in narrative:
                    keyword_score += weight
            keyword_bonus = 1.0 + (keyword_score * 0.5)
            
            # 连贯性权重
            coherence_score = self._calculate_coherence(ev, events)
            
            # 强化计数加成
            re_intensity = self.config.get("reinforcement_intensity", 1.0)
            reinforcement_boost = ev.get('reinforcement_count', 0) * 0.1 * re_intensity
            
            # 综合评分
            # 使用指数加权，系数为 1.0 时保持原样
            base_score = (importance ** w_importance) * \
                         (intensity ** w_intensity) * \
                         (date_weight ** w_time) * \
                         relevance * \
                         (keyword_bonus ** w_rare)
            return base_score * (1 + coherence_score * 0.3) * (1 + reinforcement_boost)

        return sorted(events, key=score_func, reverse=True)

    def _calculate_coherence(self, target_event: dict, all_events: List[dict]) -> float:
        """计算事件与检索结果集中其他事件的连贯性"""
        if not all_events:
            return 0.0
        
        coherence = 0.0
        
        # 与同日期事件的连贯性
        same_day_count = sum(1 for e in all_events if e['date'] == target_event['date'])
        if same_day_count > 1:
            coherence += 0.2
        
        # 与相邻日期事件的连贯性
        try:
            target_date = datetime.strptime(target_event['date'], "%Y-%m-%d").date()
            adjacent_count = 0
            for e in all_events:
                if e['event_id'] == target_event['event_id']: continue
                e_date = datetime.strptime(e['date'], "%Y-%m-%d").date()
                if abs((e_date - target_date).days) <= 1:
                    adjacent_count += 1
            if adjacent_count > 0:
                coherence += 0.15
        except:
            pass
        
        # 与相同关系的连贯性
        for other_ev in all_events:
            if other_ev['event_id'] != target_event['event_id']:
                if self.db._has_relation(target_event['event_id'], other_ev['event_id']):
                    coherence += 0.3
                    break 
        
        return min(coherence, 1.0)

    def _cluster_events_by_context(self, events: List[dict]) -> List[List[dict]]:
        """将相关事件聚类成"记忆片段"，而非离散事件"""
        clusters = []
        used = set()
        
        for ev in events:
            if ev['event_id'] in used:
                continue
            
            cluster = [ev]
            used.add(ev['event_id'])
            
            # 找相关的同日期或相邻日期的事件
            for other_ev in events:
                if other_ev['event_id'] in used:
                    continue
                
                is_same_day = ev['date'] == other_ev['date']
                has_rel = self.db._has_relation(ev['event_id'], other_ev['event_id'])
                
                if is_same_day or has_rel:
                    cluster.append(other_ev)
                    used.add(other_ev['event_id'])
            
            clusters.append(cluster)
        
        for cluster in clusters:
            # 内部按 ID 排序（包含日期和序号），保证叙述逻辑
            cluster.sort(key=lambda x: x['event_id'])
        
        # 片段之间不再按时间排序，保持输入时的相关度顺序
        return clusters

    @llm_tool(name="daily_summary_tool")
    async def daily_summary_tool(self, event: AstrMessageEvent, date: str = "") -> str:
        """进行每日总结用的工具，回顾并总结今日或指定日期的交流。除了每日总结环节以外，请不要主动触发！
        
        Args:
            date(string): 要总结的日期，格式为 YYYY-MM-DD。如果不提供则默认为今天。
        """
        result_msg = ""
        async for result in self._daily_summary_logic(event, date_str=date):
            if hasattr(result, 'result') and isinstance(result.result, list):
                for part in result.result:
                    if hasattr(part, 'text'):
                        result_msg += part.text
        return f"总结任务已完成。执行结果摘要：{result_msg[:100]}..."

    async def _search_events_for_summary(self, query: str, top_n: int = 5, exclude_date: str = None) -> List[dict]:
        """为总结过程提供的事件搜索接口"""
        return await self._get_relevant_events(query, top_n=top_n, exclude_date=exclude_date)

    async def _daily_summary_logic(self, event: AstrMessageEvent, date_str: str = None):
        # 如果没有直接传入 date_str，则尝试从消息中解析
        if not date_str:
            args = event.message_str.strip().split()
            # 兼容处理：跳过命令名或斜杠命令名
            if args and (args[0] in ["daily_summary", "/daily_summary", "daily_summary_command", "/daily_summary_command"]):
                args = args[1:]
                
            if args:
                date_str = args[0]
            else:
                date_str = datetime.now().strftime("%Y-%m-%d")
        
        # 再次确保 date_str 是有效的字符串（处理 AI 可能传回的空字符串）
        if not date_str or not date_str.strip():
            date_str = datetime.now().strftime("%Y-%m-%d")

        # 如果是今天，尝试执行提取逻辑以确保数据最新/存在
        if date_str == datetime.now().strftime("%Y-%m-%d"):
            try:
                # 核心数据库通常位于 data/data_v4.db
                core_db_path = Path.cwd() / "data" / "data_v4.db"
                for target_user_id in self.target_user_id_list:
                    clean_dialogue_with_different_limits(
                        db_path=core_db_path,
                        output_dir=self.dialog_folder,
                        username=self.username,
                        ai_name=self.ai_name,
                        platform="AstrBot",
                        target_date=date_str,
                        target_user_id=target_user_id
                    )
            except Exception as e:
                logger.error(f"提取聊天记录失败: {e}")

        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            yield event.plain_result("日期格式好像不对哦，记得用 YYYY-MM-DD 这种格式~")
            return
        
        # 寻找聊天记录文件并合并
        conversation_chunks = []
        found_any = False
        
        max_kb = self.config.get("max_dialogue_kb_per_summary", 40)
        gap_hours = self.config.get("chunk_time_gap_hours", 1.0)

        current_chunk = []
        current_size = 0.0
        last_time = None

        for target_user_id in self.target_user_id_list:
            safe_id = target_user_id.replace(":", "_") if target_user_id else ""
            dialog_file = self.dialog_folder / f"{date_str}_dialog_{safe_id}.json"
            if not dialog_file.exists():
                # 如果没找到，尝试提取一次（可能是补录历史记录）
                try:
                    core_db_path = Path.cwd() / "data" / "data_v4.db"
                    clean_dialogue_with_different_limits(
                        db_path=core_db_path,
                        output_dir=self.dialog_folder,
                        username=self.username,
                        ai_name=self.ai_name,
                        platform="AstrBot",
                        target_date=date_str,
                        target_user_id=target_user_id
                    )
                except Exception as e:
                    logger.error(f"提取聊天记录失败: {e}")
                
                # 提取后再检查一次
                if not dialog_file.exists():
                    logger.info(f"[APLR] 没找到 {date_str} ({target_user_id}) 的聊天记录，尝试路径: {dialog_file}")
                    continue

            # 读取聊天记录
            try:
                with open(dialog_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                conversations = data.get("conversations", [])
                if conversations:
                    found_any = True
                    
                    # 预估文件大小 (KB)
                    file_text_list = [f"[{m.get('timestamp', '')}] {m.get('role', 'unknown')}: {m.get('content', '')}\n" for m in conversations]
                    file_size_kb = len("".join(file_text_list).encode('utf-8')) / 1024.0
                    is_small_file = file_size_kb < (max_kb / 4.0)
                    
                    # 检查是否需要在进入新文件前切分（如果当前分段已经满了）
                    # 即使没有时间间隔，如果已经达到上限，也应该在文件边界切分，避免分段过大
                    if current_chunk and current_size >= max_kb:
                        conversation_chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_size = 0.0

                    # 添加用户 ID 标识头
                    header = f"\n=== 对话记录 ({target_user_id}) ===\n"
                    header_size = len(header.encode('utf-8')) / 1024.0
                    current_chunk.append(header)
                    current_size += header_size
                    
                    for i, msg in enumerate(conversations):
                        timestamp = msg.get('timestamp', '')
                        msg_text = file_text_list[i]
                        msg_size = len(msg_text.encode('utf-8')) / 1024.0
                        
                        time_gap_exceeded = False
                        if last_time and timestamp:
                            try:
                                curr_t = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                prev_t = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
                                if (curr_t - prev_t).total_seconds() > gap_hours * 3600:
                                    time_gap_exceeded = True
                            except:
                                pass
                        
                        # 核心逻辑：如果是小文件，则绝对不在此循环内切分（即挤一挤，不分段）
                        if not is_small_file and len(current_chunk) > 2 and (current_size + msg_size > max_kb) and time_gap_exceeded:
                            conversation_chunks.append("".join(current_chunk))
                            current_chunk = [f"\n=== 对话记录 ({target_user_id}) (续) ===\n"]
                            current_size = len(current_chunk[0].encode('utf-8')) / 1024.0
                        
                        current_chunk.append(msg_text)
                        current_size += msg_size
                        last_time = timestamp
            except Exception as e:
                logger.info(f"[APLR] 读取记录 ({target_user_id}) 的时候出了点小状况：{e}")

        if current_chunk:
            conversation_chunks.append("".join(current_chunk))

        # 在发送给总结器之前，尝试注入 thoughts 插件的中期记忆
        if conversation_chunks:
            thoughts_data_path = Path.cwd() / "data" / "plugin_data" / "astrbot_plugin_thoughts" / "interim_memory.json"
            if thoughts_data_path.exists():
                try:
                    with open(thoughts_data_path, "r", encoding="utf-8") as f:
                        interim_memories = json.load(f)
                        if isinstance(interim_memories, list) and interim_memories:
                            content = "\n".join(interim_memories)
                            interim_header = f"\n=== 中期记忆 ===\n{content}\n"
                            conversation_chunks[0] = interim_header + conversation_chunks[0]
                            logger.info(f"[APLR] 已成功注入 thoughts 插件的中期记忆到第一段总结背景中。")
                except Exception as e:
                    logger.error(f"[APLR] 读取 thoughts 中期记忆失败: {e}")

        if not found_any:
            logger.info(f"[APLR] 没找到 {date_str} 的任何聊天记录，是不是那天没说话呀？")
            return

        logger.info(f"[APLR] 正在总结 {date_str} 的点点滴滴，分成了 {len(conversation_chunks)} 段处理，稍等一下哦...")
        
        # --- 获取 LLM 提供者并初始化总结器 (新版推荐方式) ---
        try:
            # 1. 获取当前会话的 provider ID
            umo = event.unified_msg_origin
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            
            if not provider_id:
                logger.info("[APLR] 暂时连接不到大脑（LLM Provider），请检查配置。")
                return

            # 2. 定义适配 DailySummarizer 的生成函数
            async def llm_generate_func(prompt, system_prompt):
                return await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=[]
                )

            # 3. 获取基础提示词
            base_system_prompt = self.config.get("default_prompt", "") if self.config else ""
            base_user_prompt = self.config.get("user_prompts", "") if self.config else ""

            # 注入 Astrbot 自带的人格提示词
            try:
                persona = await self.context.persona_manager.get_default_persona_v3(umo=umo)
                if persona and persona.get('prompt'):
                    base_system_prompt = (persona['prompt'] + "\n" + base_system_prompt) if base_system_prompt else (persona['prompt'] + "\n")
                    logger.debug(f"[APLR] 已成功注入 Astrbot 自带的人格设定 (Persona: {persona.get('name', 'Unknown')})")
                    logger.debug(f"base sys prompt: {base_system_prompt}")
            except Exception as pe:
                logger.warning(f"[APLR] 尝试获取 Astrbot 人格设定失败: {pe}")

            # 4. 初始化总结器
            summarizer = DailySummarizer(
                llm_generate_func=llm_generate_func,
                ai_name=self.ai_name,
                base_system_prompt=base_system_prompt,
                base_user_prompt=base_user_prompt,
                search_events_func=self._search_events_for_summary
            )

            # 获取现有节点背景 (总结时放宽上限，允许描述匹配)
            full_text = "\n".join(conversation_chunks)
            nodes, _ = await self._get_nodes_context(full_text, include_description=True, max_nodes=100)
            existing_nodes_context = ""
            if nodes:
                existing_nodes_context = "\n".join([f"- {n['name']}: {n['description']}" for n in nodes])

            # 5. 异步调用总结生成
            enable_phase2 = self.config.get("enable_cross_day_relations", False)
            summary = await summarizer.generate_summary(
                conversation_chunks, 
                date_str, 
                existing_nodes_context=existing_nodes_context,
                enable_cross_day_relations=enable_phase2
            )

        except Exception as e:
            logger.error(f"[APLR] 生成总结过程中发生错误: {e}", exc_info=True)
            logger.info(f"[APLR] 生成总结过程中发生错误：{e}")
            return

        if summary is None:
            logger.info("[APLR] 诶，没能整理出有意义的总结呢")
            return

        try:
            # 存储到数据库
            self.db.insert_summary(summary)
            
            # 自动向量化当天的事件
            try:
                events_to_vec = []
                for e in summary.events:
                    events_to_vec.append({
                        "event_id": e.event_id,
                        "narrative": e.narrative
                    })
                self.vector_db.add_events(events_to_vec)
                logger.info(f"[APLR] 已自动将 {date_str} 的 {len(events_to_vec)} 个事件同步到向量数据库。")
            except Exception as ve:
                logger.error(f"[APLR] 自动向量化失败: {ve}")

            # 构建回复
            yield event.plain_result(f"✨ {date_str} 的回忆整理好啦！\n\n")
            resp = f"💭 我的感悟：{summary.daily_reflection}\n"
            resp += f"📍 我记下了 {len(summary.events)} 个印象深刻的瞬间。"
            yield event.plain_result(resp)
        except Exception as e:
            logger.exception("存入数据库失败")
            yield event.plain_result(f"[APLR] 虽然想起来了，但没能存进记忆库：{e}")

    async def terminate(self):
        """默认不开启: 当插件被卸载/停用时调用，清理模型缓存和向量数据库。不会删除记忆db文件。"""
        if 0: 
            # 1. 清理向量数据库目录
            if hasattr(self, 'vector_db_path') and self.vector_db_path.exists():
                try:
                    shutil.rmtree(self.vector_db_path)
                    logger.info(f"[APLR] 已清理向量数据库目录: {self.vector_db_path}")
                except Exception as e:
                    logger.error(f"[APLR] 清理向量数据库目录失败: {e}")

            # 2. 尝试清理模型缓存 (针对 sentence-transformers)
            try:
                embedding_model = self.config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
                # 默认路径通常格在 ~/.cache/torch/sentence_transformers/
                model_cache_base = Path.home() / '.cache' / 'torch' / 'sentence_transformers'
                # sentence-transformers 存储路径通常是模型名的下划线替换版本
                model_name_path = model_cache_base / embedding_model.replace("/", "_")
                
                if model_name_path.exists():
                    shutil.rmtree(model_name_path)
                    logger.info(f"[APLR] 已清理向量化模型缓存: {model_name_path}")
            except Exception as e:
                logger.error(f"[APLR] 清理向量化模型缓存失败: {e}")

    # --- 外部 API 接口 ---

    def is_model_ready(self) -> bool:
        """检查向量化模型是否已加载完成"""
        return hasattr(self, 'vector_db') and self.vector_db is not None and hasattr(self.vector_db, 'model')

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        供其他插件调用的 API：获取文本的向量表示。
        
        用法示例：
        reminiscence = context.get_star("local_reminiscence")
        if reminiscence and reminiscence.is_model_ready():
            vectors = reminiscence.get_embeddings(["你好", "再见"])
        """
        if not self.is_model_ready():
            logger.warning("[APLR] 有其他插件尝试调用 get_embeddings 但模型尚未就绪。")
            return []
        return self.vector_db.get_embeddings(texts)
