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

@register("local_reminiscence", "olozhika", "本地记忆插件，包含每日总结工具", "1.1.3")
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
            offline_mode=self.offline_mode
        )

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在生成回复前注入历史记忆，帮助 AI 维持长期记忆"""
        try:
            message_str = event.message_str
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
                    memory_text = "\n\n【历史记忆 - 这是你对过去几天的回忆，可以在有需要的时候参考这些信息，你并非必须在对话中提及它们】\n"
                    
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

            # 0. 提取当前用户的 Nickname
            nickname_match = re.search(r'<system_reminder>.*?Nickname:\s*([^,\n\s]+)', message_str, re.IGNORECASE)
            current_nickname = nickname_match.group(1).strip() if nickname_match else self.username
            
            # 1. 获取所有相关节点
            all_nodes = []
            seen_names = set()
            
            # 首先尝试获取用户本人节点
            if current_nickname:
                user_nodes = self.db.search_nodes(current_nickname, limit=1, include_description=False)
                if user_nodes:
                    node = user_nodes[0]
                    node['is_user'] = True # 标记为当前聊天对象
                    all_nodes.append(node)
                    seen_names.add(node['name'])

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
            hit_keyword = any(kw in message_str for kw in keywords)
            hit_prob = random.random() <= auto_recall_prob
            
            if hit_keyword or hit_prob:
                top_n = self.config.get("top_n_events", 10)
                m1 = self.config.get("m1_top_events", 3)
                m2 = self.config.get("m2_random_events", 2)
                
                search_results = self.vector_db.search_events(message_str, top_n=top_n)
                if search_results:
                    full_events = []
                    for res in search_results:
                        # 仅保留高于阈值的事件
                        if res['relevance'] >= auto_recall_threshold:
                            ev = self.db.get_event_by_id(res['event_id'])
                            if ev:
                                ev['relevance'] = res['relevance']
                                full_events.append(ev)
                    
                    if full_events:
                        ranked_events = self._rank_events(full_events, message_str)
                        selected_events = ranked_events[:m1]
                        remaining = ranked_events[m1:]
                        if remaining and m2 > 0:
                            selected_events.extend(random.sample(remaining, min(len(remaining), m2)))
                        
                        if selected_events:
                            recall_text = "\n\n【自动回想 - 你回想起了以下片段】\n"
                            for ev in selected_events:
                                recall_text += f"📅 {ev['date']}\n"
                                recall_text += f"📍 {ev['narrative']} [ID: {ev['event_id']}]\n"
                                recall_text += f"💭 {self.ai_name}感到{ev['emotion']}\n"
                                # recall_text += f"(相关度{ev.get('relevance', 0)}%)\n"
                                recall_text += "---\n"
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
                    found_nodes = self.db.search_nodes(kw, limit=2, include_description=include_description)
                
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
        """将指定日期的事件向量化并存入向量数据库。用法：/vectorize_events [日期]（格式YYYY-MM-DD）。"""
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

            summarizer = DailySummarizer(
                llm_generate_func=llm_generate_func,
                ai_name=self.ai_name,
                base_system_prompt=self.config.get("default_prompt", ""),
                base_user_prompt=self.config.get("user_prompts", "")
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
        
        return await self._get_memory_retrieval_text(query, count=count if count > 0 else None)

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

    @filter.command("write_node")
    async def write_node_command(self, event: AstrMessageEvent):
        """手动写入或更新记忆节点。用法：/write_node [节点名] [类型] [描述]"""
        msg = event.message_str.strip()
        # 移除命令名
        content = re.sub(r'^/?write_node\s*', '', msg).strip()
        
        if not content:
            yield event.plain_result("用法：/write_node [节点名] [类型] [描述]")
            return
            
        parts = content.split(maxsplit=2)
        if len(parts) < 3:
            yield event.plain_result("参数不足。用法：/write_node [节点名] [类型] [描述]\n示例：/write_node 王小美 人物 住在璃月的椰羊爱好者")
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
        search_results = self.vector_db.search_events(query, top_n=top_n)
        if not search_results:
            return "暂时没有找到相关的记忆呢。"

        # 3. 获取完整信息并合并相关度
        full_events = []
        for res in search_results:
            eid = res['event_id']
            relevance = res['relevance']
            ev = self.db.get_event_by_id(eid)
            if ev:
                ev['relevance'] = relevance # 将相关度存入事件对象
                full_events.append(ev)
        
        if not full_events:
            return "检索到了 ID 但没能从数据库找到详细信息。"

        # 4. 排序
        ranked_events = self._rank_events(full_events, query)

        # 5. 选择 M1 + M2
        selected_events = ranked_events[:m1]
        remaining = ranked_events[m1:]
        if remaining and m2 > 0:
            selected_events.extend(random.sample(remaining, min(len(remaining), m2)))

        # 6. 格式化输出
        resp = f"🔍 回想起了相关记忆：\n\n" #关于“{query}”的
        
        # 优化：直接从 selected_events 中提取 tags 并查询 nodes 表
        node_names = set()
        for ev in selected_events:
            # 获取该事件的 tags
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
                resp += f"  - {node['name']} ({node['type']}) [最后更新: {last_updated_date}]: {node['description']}\n"
            resp += "\n"

        for ev in selected_events:
            resp += f"📅 {ev['date']}\n"
            resp += f"📍 {ev['narrative']} [ID: {ev['event_id']}]\n"
            resp += f"💭 {self.ai_name}感到{ev['emotion']}\n"
            resp += f"(事件相关度{ev.get('relevance', 0)}%)\n"
            resp += "---\n"
        
        if self.config.get("encourage_deep_recall", False):
            resp += "\n（注：若想进一步回忆某个事件的细节 or 感想，可以使用 `recall_event_reflection_tool` 并传入对应的 事件ID来深度回想；如果觉得回忆还不够充分，可以使用 `recall_memory_tool` 对刚想起的片段进行联想）"
        return resp

    def _rank_events(self, events: List[dict], query: str = "") -> List[dict]:
        """按照距今日期差、事件重要性、情感强度以及关键词匹配度进行综合排序"""
        now = datetime.now().date()
        
        # 预处理：计算局部关键词权重 (Simple Local IDF)
        # 这样可以自动识别出哪些词是“罕见”的（如“相声”），哪些是“常见”的（如“Lanya”）
        word_weights = {}
        if query:
            # 使用 jieba 进行分词，并过滤掉单字（通常是虚词）
            # 同时保留英文单词和数字
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
                    # IDF 公式：log(总文档数 / (包含该词的文档数 + 1)) + 1
                    # 这样罕见词的权重会显著高于常见词
                    count = doc_counts.get(term, 0)
                    word_weights[term] = math.log(num_docs / (count + 1.0)) + 1.0

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
            
            # 关键词加成 (Keyword Bonus)
            keyword_score = 0.0
            narrative = ev.get('narrative', '')
            for term, weight in word_weights.items():
                if term in narrative:
                    keyword_score += weight
            
            # 关键词加成系数：1.0 + (匹配词的总权重 / 10)
            # 这样既保留了向量搜索的泛化能力，又给了精准匹配显著的加成
            keyword_bonus = 1.0 + (keyword_score * 0.5)
            
            # 综合评分
            return importance * intensity * date_weight * relevance * keyword_bonus

        return sorted(events, key=score_func, reverse=True)

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
        conversation_text = ""
        found_any = False
        
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
                    conversation_text += f"\n=== 对话记录 ({target_user_id}) ===\n"
                    for msg in conversations:
                        timestamp = msg.get('timestamp', '')
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        conversation_text += f"[{timestamp}] {role}: {content}\n"
            except Exception as e:
                logger.info(f"[APLR] 读取记录 ({target_user_id}) 的时候出了点小状况：{e}")

        if not found_any:
            logger.info(f"[APLR] 没找到 {date_str} 的任何聊天记录，是不是那天没说话呀？")
            return

        logger.info(f"[APLR] 正在总结 {date_str} 的点点滴滴，稍等一下哦...")
        
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

            # 4. 初始化总结器
            summarizer = DailySummarizer(
                llm_generate_func=llm_generate_func,
                ai_name=self.ai_name,
                base_system_prompt=base_system_prompt,
                base_user_prompt=base_user_prompt
            )

            # 获取现有节点背景 (总结时放宽上限，允许描述匹配)
            nodes, _ = await self._get_nodes_context(conversation_text, include_description=True, max_nodes=100)
            existing_nodes_context = ""
            if nodes:
                existing_nodes_context = "\n".join([f"- {n['name']}: {n['description']}" for n in nodes])

            # 5. 异步调用总结生成
            summary = await summarizer.generate_summary(conversation_text, date_str, existing_nodes_context=existing_nodes_context)

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
