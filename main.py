import os
import asyncio
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
from .memory_consolidation import ThematicConsolidator
from .chat_history_extract import clean_dialogue_with_different_limits
import random
import shutil
import subprocess
import sys
import numpy as np

@register("local_reminiscence", "olozhika", "基于定时总结和向量化的本地记忆插件", "1.3.5")
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
            import sklearn
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
                    # 2. 再安装 sentence-transformers, chromadb, scikit-learn
                    pip_cmd = [sys.executable, "-m", "pip", "install", "sentence-transformers", "chromadb", "scikit-learn"]
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

        data_dir = StarTools.get_data_dir("astrbot_plugin_local_reminiscence")
        
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
            ai_name=self.ai_name,
            idle_timeout=self.config.get("model_idle_timeout", -1)
        )
        
        # 启动背景检测任务
        asyncio.create_task(self._model_idle_check_loop())
        asyncio.create_task(self._auto_summary_check_loop())
        
        # 延迟初始化固化器，因为需要 LLM 函数
        self.consolidator = None

    def _parse_cron_time(self, cron_str: str) -> tuple[int, int]:
        """从 cron 字符串中解析出小时和分钟。
        假定格式为 'm h * * *' (例如 '0 4 * * *')。如果是其他复杂/不合法格式，兜底返回 (4, 0) 代表凌晨4点。
        """
        try:
            parts = cron_str.strip().split()
            if len(parts) >= 2:
                minute = int(parts[0])
                hour = int(parts[1])
                if 0 <= minute < 60 and 0 <= hour < 24:
                    return hour, minute
        except Exception as e:
            logger.error(f"[APLR] 解析 cron 发生异常 '{cron_str}': {e}")
        return 4, 0  # 默认凌晨 4:00

    def _get_logical_date(self, dt: datetime) -> str:
        """获取给定时间的逻辑日期"""
        boundary_conf = self.config.get("day_boundary_config", {})
        boundary_cron = boundary_conf.get("boundary_cron", "0 4 * * *")
        h, m = self._parse_cron_time(boundary_cron)
        
        # 构造同 calendar 天的 boundary limit
        boundary_dt = dt.replace(hour=h, minute=m, second=0, microsecond=0)
        from datetime import timedelta
        if dt < boundary_dt:
            t_start = boundary_dt - timedelta(days=1)
        else:
            t_start = boundary_dt
        t_mid = t_start + timedelta(hours=12)
        return t_mid.strftime("%Y-%m-%d")

    def _get_completed_logical_date(self) -> str:
        """获取刚刚结束的逻辑日期。如果是自动总结，应该针对此日期进行。"""
        # 稍微拨回 5 分钟
        from datetime import timedelta
        past_time = datetime.now() - timedelta(minutes=5)
        return self._get_logical_date(past_time)

    def _is_in_exclusion_window(self, dt: datetime) -> bool:
        """检查给定时间是否在 [boundary_time, boundary_time + 10min) 的排除窗口内"""
        boundary_conf = self.config.get("day_boundary_config", {})
        boundary_cron = boundary_conf.get("boundary_cron", "0 4 * * *")
        h, m = self._parse_cron_time(boundary_cron)
        
        # 计算 boundary 对应的分钟数
        boundary_mins = h * 60 + m
        # 计算当前时间对应的分钟数
        dt_mins = dt.hour * 60 + dt.minute
        
        diff = (dt_mins - boundary_mins) % 1440
        return 0 <= diff < 10

    async def _auto_summary_check_loop(self):
        """自动后台任务（每日总结和会话结束重置）触发检测循环（每分钟检查一次）"""
        while True:
            try:
                await asyncio.sleep(60)
                
                # 读取配置
                boundary_conf = self.config.get("day_boundary_config", {})
                auto_summary = boundary_conf.get("auto_summary_enabled", False)
                auto_end = boundary_conf.get("auto_end_session", False)
                if not auto_summary and not auto_end:
                    continue
                
                boundary_cron = boundary_conf.get("boundary_cron", "0 4 * * *")
                h, m = self._parse_cron_time(boundary_cron)
                now = datetime.now()
                
                # 1. 自动每日总结 (+1 分钟)
                if auto_summary:
                    target_h = h
                    target_m = m + 1
                    if target_m >= 60:
                        target_m %= 60
                        target_h = (target_h + 1) % 24
                    
                    if now.hour == target_h and now.minute == target_m:
                        logger.info(f"[APLR] 触发自动每日总结：当前时间符合{boundary_cron} + 1min 时间点 ({target_h:02d}:{target_m:02d})")
                        target_date = self._get_completed_logical_date()
                        await self._run_automatic_daily_summary(target_date)
                
                # 2. 自动关闭所有激活会话 (+9 分钟)
                if auto_end:
                    target_h = h
                    target_m = m + 9
                    if target_m >= 60:
                        target_m %= 60
                        target_h = (target_h + 1) % 24
                    
                    if now.hour == target_h and now.minute == target_m:
                        logger.info(f"[APLR] 触发自动结束/重置上下文会话：当前时间符合{boundary_cron} + 9min 时间点 ({target_h:02d}:{target_m:02d})")
                        await self._close_all_active_sessions()
            except Exception as e:
                logger.error(f"[APLR] 自动后台任务检测循环发生异常: {e}")

    async def _close_all_active_sessions(self):
        """关闭/重置所有当前有记录的活页 Session (通过清空内存缓存和 sel_conv_id 实现下一次说话自动切换新会话)"""
        logger.info("[APLR] 自动结束/重置所有激活中的对话上下文任务开始运行...")
        try:
            from astrbot.core import sp
            from astrbot.core.utils.active_event_registry import active_event_registry
            
            conv_mgr = self.context.conversation_manager
            
            # 1. 搜集所有当前的会话 UMO (除了拉取 DB, 也需要包含内存中那些未同步或在活动中的 session)
            convs, total = await conv_mgr.db.get_all_conversations(page=1, page_size=2000)
            
            # 使用集合去重
            umos = {conv.user_id for conv in convs if conv.user_id}
            
            # 把内存中 session_conversations 里的 session UMO 也合并进来
            for umo in list(conv_mgr.session_conversations.keys()):
                umos.add(umo)
                
            closed_count = 0
            for umo in umos:
                if not umo:
                    continue
                
                # 获取当前的对话 ID 并进行状态验证
                curr_cid = await conv_mgr.get_curr_conversation_id(umo)
                if not curr_cid:
                    continue
                
                # 查询当前对话的详细内容，只有存在内容(不为空)的活跃会话才会被处理重置
                conv = await conv_mgr.db.get_conversation_by_id(curr_cid)
                if conv and conv.content and len(conv.content) > 0:
                    # 停止对该 UMO 正在后台运行的所有事件
                    active_event_registry.request_agent_stop_all(umo)
                    
                    # 关键操作：清除内存对该 UMO 当前 selected 对话的缓存指向
                    conv_mgr.session_conversations.pop(umo, None)
                    
                    # 关键操作：移去偏好设置，切断关联。下次说话触发 _get_session_conv 会由于 get_curr_conversation_id(umo) 为空而触发全新 new_conversation
                    await sp.session_remove(umo, "sel_conv_id")
                    
                    closed_count += 1
                    
            logger.info(f"[APLR] 自动结束/重置所有旧会话上下文执行完毕。共安全移除了 {closed_count} 个活跃会话的当前指针 (历史数据完好保留在数据库中)。")
        except Exception as e:
            logger.error(f"[APLR] 自动结束/重置所有会话上下文时发生异常: {e}", exc_info=True)

    async def _run_automatic_daily_summary(self, target_date: str):
        logger.info(f"[APLR] 开始执行自动每日总结，日期: {target_date}")
        try:
            async for result in self._daily_summary_logic(event=None, date_str=target_date):
                # 消费生成器
                pass
            logger.info(f"[APLR] 自动每日总结执行完毕，日期: {target_date}")
        except Exception as e:
            logger.error(f"[APLR] 自动每日总结执行遭遇异常: {e}", exc_info=True)

    async def _model_idle_check_loop(self):
        """向量模型闲置检测循环"""
        while True:
            try:
                await asyncio.sleep(600) # 每 10 分钟检查一次
                if hasattr(self, 'vector_db'):
                    self.vector_db.check_and_unload_model()
            except Exception as e:
                logger.error(f"[APLR] 向量模型闲置检测任务异常: {e}")

    def _is_session_matching(self, unified_id: str) -> bool:
        """检查会话ID是否匹配监测配置"""
        if not unified_id:
            return False
        target_list = list(self.target_user_id_list) if isinstance(self.target_user_id_list, list) else [self.target_user_id_list]
        if len(target_list) == 1 and str(target_list[0]).strip().lower() == "all":
            return True
        return unified_id in target_list

    def _get_effective_user_ids(self, date_str: str = None) -> List[str]:
        """获取需要导出的实际会话ID列表，当配置为['all']时自动检索所有用户的ID"""
        target_list = list(self.target_user_id_list) if isinstance(self.target_user_id_list, list) else [self.target_user_id_list]
        if len(target_list) == 1 and str(target_list[0]).strip().lower() == "all":
            effective_user_ids = []
            
            # 1. 扫描本地文件夹中的现有 JSON 会话记录（针对该日期已录制的情况）
            if date_str:
                try:
                    for filepath in self.dialog_folder.glob(f"{date_str}_dialog_*.json"):
                        filename = filepath.name
                        prefix_len = len(f"{date_str}_dialog_")
                        suffix_len = len(".json")
                        safe_id = filename[prefix_len:-suffix_len]
                        if safe_id and safe_id not in effective_user_ids:
                            effective_user_ids.append(safe_id)
                except Exception as e:
                    logger.error(f"[APLR] 扫描本地 JSON 发现用户ID失败: {e}")
            
            # 2. 从核心数据库获取所有用户 ID（用于未开启实时录制但需提取全部的情况）
            try:
                import sqlite3
                core_db_path = Path.cwd() / "data" / "data_v4.db"
                if core_db_path.exists():
                    conn = sqlite3.connect(str(core_db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT user_id FROM conversations")
                    db_user_ids = [row[0] for row in cursor.fetchall() if row[0]]
                    conn.close()
                    for db_uid in db_user_ids:
                        if db_uid not in effective_user_ids:
                            effective_user_ids.append(db_uid)
            except Exception as e:
                logger.error(f"[APLR] 从核心数据库获取所有用户ID失败: {e}")
                
            if not effective_user_ids:
                effective_user_ids = ["all"]
            return effective_user_ids
        else:
            return target_list

    def _append_to_realtime_log(self, unified_id: str, role: str, content: str):
        """将消息追加到实时日志文件 (JSON)"""
        try:
            if not self.config.get("realtime_recording", False):
                return
            
            now = datetime.now()
            if self._is_in_exclusion_window(now):
                return
            
            date_str = self._get_logical_date(now)
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            
            safe_id = unified_id.replace(":", "_") if unified_id else "unknown"
            folder = self.dialog_folder
            folder.mkdir(parents=True, exist_ok=True)
            dialog_file = folder / f"{date_str}_dialog_{safe_id}.json"
            
            # 使用 JSON 覆盖模式，抗文件破损
            data = {"conversations": []}
            if dialog_file.exists():
                try:
                    with open(dialog_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    logger.error(f"[APLR] 读取实时 JSON 失败，可能存在破损 (已备份容错): {e}")
                    try:
                        dialog_file.rename(dialog_file.with_suffix('.json.corrupted'))
                    except:
                        pass
                    data = {"conversations": []}
            
            if not isinstance(data.get("conversations"), list):
                data["conversations"] = []

            data["conversations"].append({
                "timestamp": timestamp,
                "role": role,
                "content": content
            })
            
            # 使用原子层级写入：写入 temp file 成功后直接原子替换
            temp_file = dialog_file.with_suffix('.json.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                temp_file.replace(dialog_file)
            except Exception as e:
                logger.error(f"[APLR] 写入实时 JSON 失败: {e}")
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
        except Exception as e:
            logger.error(f"[APLR] _append_to_realtime_log 遭遇不可期异常 (不影响聊天进程): {e}")

    def _get_unified_id(self, event: any) -> str:
        """从各种可能的事件对象中安全获取 unified_id"""
        # 如果是包装事件 (OnLLMRequestEvent, OnLLMResponseEvent, etc.)
        actual_event = getattr(event, 'event', event)
        
        # 尝试直接获取属性
        uid = getattr(actual_event, 'unified_id', None)
        if uid:
            return uid
            
        # 尝试从消息源获取
        uid = getattr(actual_event, 'unified_msg_origin', None)
        if uid:
            return uid
            
        # 尝试调用方法
        if hasattr(actual_event, 'get_unified_id'):
            try:
                return actual_event.get_unified_id()
            except:
                pass

        # 尝试从 message_obj 构造 (AstrBot v4 常见结构)
        try:
            msg_obj = getattr(actual_event, 'message_obj', None)
            if msg_obj:
                platform = getattr(msg_obj, 'platform', '')
                sender = getattr(msg_obj, 'sender', None)
                if sender and hasattr(sender, 'user_id'):
                    return f"{platform}:{sender.user_id}"
        except:
            pass

        return None

    def _get_sender_nickname(self, event: any) -> str:
        """多重回退获取发送者昵称"""
        actual_event = getattr(event, 'event', event)
        
        # 1. 尝试 get_sender_name()
        if hasattr(actual_event, 'get_sender_name'):
            try:
                name = actual_event.get_sender_name()
                if name: return name
            except:
                pass
        
        # 2. 尝试从 message_obj 获取
        try:
            msg_obj = getattr(actual_event, 'message_obj', None)
            if msg_obj and hasattr(msg_obj, 'sender'):
                sender = msg_obj.sender
                # 尝试常见字段
                for attr in ['nickname', 'name', 'card', 'user_id']:
                    val = getattr(sender, attr, None)
                    if val: return str(val)
        except:
            pass
            
        # 3. 实在没有就用配置里的
        return self.username or "User"

    @filter.on_llm_request()
    async def on_llm_request(self, event: any, *args, **kwargs):
        """在生成回复前注入历史记忆，帮助 AI 维持长期记忆"""
        # 如果未开启实时记录且不需要注入逻辑，理论上可以跳过大半，但注入逻辑是核心功能，所以只在记录处检查。
        
        # 记录用户消息（实时记录模式）
        try:
            unified_id = self._get_unified_id(event) or "system"
            if self.config.get("realtime_recording", False):
                if self._is_session_matching(unified_id):
                    actual_event = getattr(event, 'event', event)
                    nickname = self._get_sender_nickname(event)
                    text = actual_event.get_plain_text() if hasattr(actual_event, 'get_plain_text') else getattr(actual_event, 'message_str', '')
                    if text:
                        self._append_to_realtime_log(unified_id, nickname, text)
        except Exception as e:
            logger.error(f"[APLR] 实时记录用户消息异常 (不影响聊天): {e}")
        
        unified_id = self._get_unified_id(event) or "system"
        # 提取 req (ProviderRequest)
        req = kwargs.get('req') or (args[0] if args and isinstance(args[0], ProviderRequest) else None)
        if not req:
            # 尝试从 args 中找
            for arg in args:
                if isinstance(arg, ProviderRequest):
                    req = arg
                    break
        
        if not req:
            return

        try:
            actual_event = getattr(event, 'event', event)
            message_str = getattr(actual_event, 'message_str', '')
            if not message_str:
                return
            
            # 1. 检测是否为新对话开启
            uid = getattr(actual_event, 'unified_msg_origin', None)
            if not uid:
                uid = unified_id
            
            if not uid:
                return # 无法识别会话来源，跳过注入
                
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(uid)
            
            is_new_session = False
            if curr_cid is None:
                is_new_session = True
            else:
                conversation: Conversation = await conv_mgr.get_conversation(uid, curr_cid)
                # 下面这行这么写纯粹是因为作者olozhika不熟悉json格式...
                history_json_empty = False
                if conversation and conversation.history:
                    try:
                        history_json_empty = (json.loads(conversation.history) == [])
                    except Exception:
                        history_json_empty = True
                if not conversation or not conversation.history or conversation.history == '[]' or history_json_empty:
                    is_new_session = True
            
            # 如果是新对话开启，则注入历史摘要
            if is_new_session:
                # 2. 获取最近3天的总结
                summaries = self.db.get_summaries(3)
                # 3. 获取最近7天的感悟
                reflections = self.db.get_reflections(7)
                
                if summaries or reflections:
                    memory_text = "\n\n【历史记忆 - 对过去几天的回忆】\n"
                    
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
                                    memory_text += "注：若想进一步回忆某个事件的细节或感想，可以使用 `deep_recall_tool` 并传入对应的 事件ID、主题ID或日期来深度回想；如果觉得回忆还不够充分，可以使用 `recall_memory_tool` 对刚想起的片段进行联想）\n"
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

            # 0. 提取当前用户的 Nickname 与群名称 Group Name (增强版多路融合提取逻辑)
            current_nickname = None
            current_group_name = None
            
            actual_event = getattr(event, 'event', event)
            message_obj = getattr(actual_event, 'message_obj', None)
            
            # 路径 A：优先从 AstrBot 原生的 message_obj 中直接提取其内存化实体属性（极度精准，杜绝文本正则表达式的失误）
            if message_obj:
                sender = getattr(message_obj, 'sender', None)
                if sender:
                    for attr in ['nickname', 'card', 'name']:
                        val = getattr(sender, attr, None)
                        if val and str(val).strip():
                            current_nickname = str(val).strip()
                            break
                    if not current_nickname and hasattr(sender, 'user_id'):
                        current_nickname = str(sender.user_id).strip()
                        
                group = getattr(message_obj, 'group', None)
                if group and hasattr(group, 'group_name'):
                    g_name = getattr(group, 'group_name', None)
                    if g_name and str(g_name).strip():
                        current_group_name = str(g_name).strip()

            # 路径 B：辅助/兜底，多阶全文扫描含有 <system_reminder> 标签或普通结构文本的字段
            texts_to_search = []
            if message_str:
                texts_to_search.append(message_str)
                
            if req:
                # 扫描 extra_user_content_parts
                extra_parts = getattr(req, 'extra_user_content_parts', [])
                if isinstance(extra_parts, list):
                    for part in extra_parts:
                        part_text = getattr(part, 'text', '') or (part.get('text', '') if isinstance(part, dict) else '')
                        if part_text:
                            texts_to_search.append(part_text)
                
                # 扫描 user_content_parts
                user_parts = getattr(req, 'user_content_parts', [])
                if isinstance(user_parts, list):
                    for part in user_parts:
                        part_text = getattr(part, 'text', '') or (part.get('text', '') if isinstance(part, dict) else '')
                        if part_text:
                            texts_to_search.append(part_text)

            # 解析并提取
            for text_val in texts_to_search:
                if not isinstance(text_val, str):
                    continue
                tag_match = re.search(r'<system_reminder>(.*?)</system_reminder>', text_val, re.IGNORECASE | re.DOTALL)
                if tag_match:
                    tag_content = tag_match.group(1)
                    if not current_nickname:
                        nick_match = re.search(r'Nickname:\s*([^\n\r,<>]+)', tag_content, re.IGNORECASE)
                        if nick_match:
                            current_nickname = nick_match.group(1).strip()
                            logger.debug(f"[APLR] 从标签中提取 Nickname 成功: {current_nickname}")
                    if not current_group_name:
                        group_match = re.search(r'Group name:\s*([^\n\r,<>]+)', tag_content, re.IGNORECASE)
                        if group_match:
                            current_group_name = group_match.group(1).strip()
                            logger.debug(f"[APLR] 从标签中提取 Group Name 成功: {current_group_name}")

                # 全文正则表达式兜底（防止未包裹在 system_reminder 标签中）
                if not current_nickname:
                    nick_match = re.search(r'Nickname:\s*([^\n\r,<>]+)', text_val, re.IGNORECASE)
                    if nick_match:
                        current_nickname = nick_match.group(1).strip()
                if not current_group_name:
                    group_match = re.search(r'Group name:\s*([^\n\r,<>]+)', text_val, re.IGNORECASE)
                    if group_match:
                        current_group_name = group_match.group(1).strip()

            if not current_nickname:
                logger.debug(f"[APLR] 无法从消息中提取 Nickname")
            else:
                logger.info(f"[APLR] 融合模块提取的当前对话用户昵称: {current_nickname}")
            if not current_group_name:
                logger.debug(f"[APLR] 无法从消息中提取 Group Name")
            else:
                logger.info(f"[APLR] 融合模块提取的当前对话群聊名称: {current_group_name}")
            
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
                
                # 双向/逆向模糊匹配备用方案：如果正面查找未中，尝试在数据库中寻找其名称被包含于 current_nickname 中的节点
                if not user_nodes:
                    try:
                        with self.db._get_conn() as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT * FROM nodes 
                                WHERE LENGTH(name) >= 2 AND ? LIKE '%' || LOWER(name) || '%'
                                ORDER BY LENGTH(name) DESC, last_updated DESC
                                LIMIT 1
                            """, (current_nickname.lower(),))
                            row = cursor.fetchone()
                            if row:
                                user_nodes = [dict(row)]
                                logger.debug(f"[APLR] 触发逆向匹配，找到用户节点: {user_nodes[0]['name']}")
                    except Exception as e:
                        logger.warning(f"[APLR] 搜索用户节点逆向匹配时出错: {e}")

                if user_nodes:
                    node = user_nodes[0]
                    logger.debug(f"[APLR] 找到用户节点: {node['name']} (ID: {node['id']})")
                    node['is_user'] = True # 标记为当前聊天对象
                    all_nodes.append(node)
                    seen_names.add(node['name'])
                else:
                    logger.debug(f"[APLR] 未找到用户节点: {current_nickname}")

            # 其次尝试获取群聊空间节点
            if current_group_name:
                logger.debug(f"[APLR] 正在搜索群聊节点: {current_group_name}")
                group_nodes = self.db.search_nodes(current_group_name, limit=1, include_description=False)
                
                # 双向/逆向模糊匹配备用方案：如果正面查找未中，尝试在数据库中寻找其名称被包含于 current_group_name 中的节点
                if not group_nodes:
                    try:
                        with self.db._get_conn() as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT * FROM nodes 
                                WHERE LENGTH(name) >= 2 AND ? LIKE '%' || LOWER(name) || '%'
                                ORDER BY LENGTH(name) DESC, last_updated DESC
                                LIMIT 1
                            """, (current_group_name.lower(),))
                            row = cursor.fetchone()
                            if row:
                                group_nodes = [dict(row)]
                                logger.debug(f"[APLR] 触发逆向匹配，找到群聊节点: {group_nodes[0]['name']}")
                    except Exception as e:
                        logger.warning(f"[APLR] 搜索群聊节点逆向匹配时出错: {e}")

                if group_nodes:
                    node = group_nodes[0]
                    if node['name'] not in seen_names:
                        logger.debug(f"[APLR] 找到群聊节点: {node['name']} (ID: {node['id']})")
                        node['is_group'] = True # 标记为当前群聊空间
                        all_nodes.append(node)
                        seen_names.add(node['name'])
                else:
                    logger.debug(f"[APLR] 未找到群聊节点: {current_group_name}")

            # 获取消息相关的节点
            msg_nodes, _ = await self._get_nodes_context(message_str, include_description=False, limit_per_kw=1)
            for node in msg_nodes:
                if node['name'] not in seen_names:
                    all_nodes.append(node)
                    seen_names.add(node['name'])

            if all_nodes:
                context_parts = []
                for n in all_nodes:
                    last_updated_date = n['last_updated'].split(' ')[0] if n['last_updated'] else "未知"
                    context_parts.append(f"📌 {n['name']}，{n['description']}。(最后更新: {last_updated_date})")
                
                injection_text = f"\n\n【记忆节点背景 - 对提及实体及当前聊天对象的已知认知】\n" + "\n\n".join(context_parts) + "\n"
                req.system_prompt += injection_text
                logger.info(f"[APLR] 自动注入了 {len(all_nodes)} 个记忆节点: {', '.join([n['name'] for n in all_nodes])}")

            # 0. 自动触发记忆检索 (语义匹配) - 放在新会话注入后面，且非新会话也会触发
            keywords = ["之前", "记得", "回忆", "想起", "以前", "过去", "曾经"]
            
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
                                recall_text += "（注：若想进一步回忆某个事件的细节或感想，可以使用 `deep_recall_tool` 并传入对应的 事件ID、主题ID或日期来深度回想；如果觉得回忆还不够充分，可以使用 `recall_memory_tool` 对刚想起的片段进行联想）"
                            req.system_prompt += recall_text
                            trigger_reason = "关键词" if hit_keyword else "概率"
                            logger.info(f"[APLR] 自动触发记忆检索({trigger_reason})，注入了 {len(selected_events)} 条相关记忆")
            
        except Exception as e:
            logger.error(f"[APLR] 注入历史记忆失败: {e}", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(self, event: any, *args, **kwargs):
        """记录 AI 的回复（实时模式）"""
        try:
            if not self.config.get("realtime_recording", False):
                return
            
            if not event:
                return
            unified_id = self._get_unified_id(event) or "system"
            if not self._is_session_matching(unified_id):
                return
                
            ai_name = self.ai_name
            
            # 优先使用传入的 resp，如果没有则尝试从事件对象中获取（部分版本兼容）
            resp = kwargs.get('resp') or (args[0] if args else getattr(event, 'resp', None))
            if not resp:
                return
            
            actual_event = getattr(event, 'event', event)
            is_cron = (actual_event.__class__.__name__ == "CronMessageEvent")
            
            # 尝试提取思考过程 (支持标准 reasoning_content 属性，以及一些带有思考的 extra 思考)
            think_content = ""
            if hasattr(resp, 'reasoning_content') and resp.reasoning_content:
                think_content = resp.reasoning_content
            elif hasattr(resp, 'extra') and isinstance(resp.extra, dict):
                # 不同 provider 的字段名可能不同
                think_content = resp.extra.get('think', '') or resp.extra.get('reasoning', '')
                
            if is_cron:
                # 3. 对于cron job中的AI说话和reasoning_content其实都是思考，在这种情况下可以把reasoning_content记作深层思考，其他内容记做浅层思考
                if think_content:
                    self._append_to_realtime_log(unified_id, ai_name, f"(深层思考: {think_content.strip()})")
                
                text = getattr(resp, 'completion_text', '')
                if text:
                    text = text.strip()
                    if text and not text.startswith("I finished this job"):
                        self._append_to_realtime_log(unified_id, ai_name, f"(浅层思考: {text})")
            else:
                if think_content:
                    self._append_to_realtime_log(unified_id, ai_name, f"(思考: {think_content.strip()})")
                
                # 记录主干回复
                text = getattr(resp, 'completion_text', '')
                if text:
                    text = text.strip()
                    # 过滤掉 APLR 内部可能产生的空回复或标记
                    if text and not text.startswith("I finished this job"):
                        self._append_to_realtime_log(unified_id, ai_name, text)
        except Exception as e:
            logger.error(f"[APLR] 实时记录 AI 回复异常 (不影响聊天): {e}")

    @filter.on_llm_tool_respond()
    async def on_llm_tool_respond(self, event: any, *args, **kwargs):
        """记录工具调用活动（实时模式）"""
        try:
            if not self.config.get("realtime_recording", False):
                return
            
            if not event:
                return
            unified_id = self._get_unified_id(event) or "system"
            if not self._is_session_matching(unified_id):
                return
                
            ai_name = self.ai_name
            
            # 优先使用单独传参，否则尝试从事件对象或 kwargs 获取
            # args[0] 是 tool_name, args[1] 是 args, args[2] 是 result (如果是通过位置参数传递)
            tool_obj = kwargs.get('tool_name') or (args[0] if len(args) > 0 else getattr(event, 'tool_name', 'unknown'))
            if hasattr(tool_obj, 'name'):
                tool_name = tool_obj.name
            else:
                tool_name = str(tool_obj)
                
            tool_args = kwargs.get('args') or (args[1] if len(args) > 1 else getattr(event, 'args', {}))
            result = kwargs.get('result') or (args[2] if len(args) > 2 else getattr(event, 'result', ''))

            # 记录调用描述
            try:
                args_str = json.dumps(tool_args, ensure_ascii=False)
            except:
                args_str = str(tool_args)
                
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            call_desc = f"(操作: 调用函数: {tool_name}({args_str}))"
            self._append_to_realtime_log(unified_id, ai_name, call_desc)
            
            # 记录结果中的异常情况
            is_error = None
            
            # 1. 企图通过属性判断
            for attr in ['is_error', 'isError']:
                if hasattr(result, attr):
                    val = getattr(result, attr)
                    if isinstance(val, bool):
                        is_error = val
                        break
                        
            # 2. 企图通过字典键判断
            if is_error is None and isinstance(result, dict):
                for k in ['is_error', 'isError']:
                    if k in result:
                        val = result[k]
                        if isinstance(val, bool):
                            is_error = val
                            break
                            
            # 3. 企图通过序列化字符串特征判断 (特别针对 MCP CallToolResult)
            res_str = str(result)
            if is_error is None:
                if "isError=False" in res_str or "is_error=False" in res_str:
                    is_error = False
                elif "isError=True" in res_str or "is_error=True" in res_str:
                    is_error = True

            if is_error is False:
                # 明确为成功，不记录失败
                pass
            elif is_error is True:
                # 明确为失败
                self._append_to_realtime_log(unified_id, ai_name, "(操作失败: Error)")
            else:
                # 兜底：无明确 is_error 标识时，通过正则做模糊判断，并清洗掉 status / flag 字样
                clean_res_str = res_str
                clean_res_str = re.sub(r'is_?error\s*=\s*\w+', '', clean_res_str, flags=re.IGNORECASE)
                
                error_re = re.compile(r"(error|failed|exception|超时|失败|not found|insufficient|denied)", re.IGNORECASE)
                match = error_re.search(clean_res_str)
                if match:
                    keyword = match.group(0).strip()
                    self._append_to_realtime_log(unified_id, ai_name, f"(操作失败: {keyword})")

            # 4. 特殊处理 send_message_to_user 工具：记录 AI 发送给用户的真实话语
            if tool_name == "send_message_to_user" and not (is_error is True):
                try:
                    target_session = tool_args.get("session")
                    target_unified_id = str(target_session) if target_session else unified_id
                    
                    if self._is_session_matching(target_unified_id):
                        msg_parts = []
                        messages_list = tool_args.get("messages", [])
                        if isinstance(messages_list, list):
                            for msg in messages_list:
                                if not isinstance(msg, dict):
                                    continue
                                type_ = str(msg.get("type", "")).lower()
                                if type_ == "plain":
                                    msg_parts.append(msg.get("text", ""))
                                elif type_ == "image":
                                    msg_parts.append("[图片]")
                                elif type_ == "record":
                                    msg_parts.append("[语音消息]")
                                elif type_ == "file":
                                    name = msg.get("text") or (os.path.basename(msg.get("path")) if msg.get("path") else "") or "file"
                                    msg_parts.append(f"[文件: {name}]")
                                elif type_ == "mention_user":
                                    msg_parts.append(f"@{msg.get('mention_user_id', '')}")
                        
                        sent_text = "".join(msg_parts).strip()
                        if sent_text and not res_str.strip().startswith("error:"):
                            self._append_to_realtime_log(target_unified_id, ai_name, sent_text)
                except Exception as sexc:
                    logger.error(f"[APLR] 记录 send_message_to_user 的内容时出错: {sexc}")
        except Exception as e:
            logger.error(f"[APLR] 实时记录工具活动异常 (不影响聊天): {e}")

    async def _get_nodes_context(self, text: str, include_description: bool = False, max_nodes: int = 8, limit_per_kw: int = 1) -> tuple[list[dict], list[str]]:
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
                    found_nodes = self.db.search_nodes(kw, limit=limit_per_kw, include_description=include_description)
                
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

    async def _get_nodes_for_summary(self, text: str, include_username: bool = False) -> list[dict]:
        """专门为总结/提取环节优化的节点获取逻辑"""
        
        # 获取配置
        dr_nodes_config = self.config.get("dailyreview_nodes", {})
        max_ref = dr_nodes_config.get("max_reference_nodes", 20)
        
        def add_special_nodes(current_nodes):
            res = list(current_nodes)
            important_names = []
            if self.ai_name: important_names.append(self.ai_name)
            if include_username and self.username: important_names.append(self.username)
            
            if important_names:
                special_nodes = self.db.get_nodes_by_names(important_names)
                seen_names = {n['name'] for n in res}
                for sn in special_nodes:
                    if sn['name'] not in seen_names:
                        res.append(sn)
                        seen_names.add(sn['name'])
            return res

        # 1. 默认检索 1 个 (limit_per_kw=1)
        nodes1_raw, _ = await self._get_nodes_context(text, include_description=True, max_nodes=100, limit_per_kw=1)
        nodes1_final = add_special_nodes(nodes1_raw)
        
        # 2. 假如去重后记忆节点数目 < 10 个，再试试检索 2 个
        if len(nodes1_raw) < 10:
            nodes2_raw, _ = await self._get_nodes_context(text, include_description=True, max_nodes=100, limit_per_kw=2)
            nodes2_final = add_special_nodes(nodes2_raw)
            
            # 3. 最终假如检索数大于 1 (即使用 limit=2)，一定要让最终给的记忆节点数目小于配置上限
            if len(nodes2_final) < max_ref:
                return nodes2_final
            # 否则回退到只检索 1 个的结果
            return nodes1_final
        
        return nodes1_final

    @filter.command("daily_summary_command")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def daily_summary_command(self, event: AstrMessageEvent):
        """进行每日总结用的工具，回顾并总结今日或指定日期的交流。用法：/daily_summary_command [日期]（可选，默认今天，格式YYYY-MM-DD）。人类专用版！（仅管理员）"""
        async for result in self._daily_summary_logic(event):
            yield result

    @filter.command_group("APLR_maintenance")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def aplr_maintenance_group(self):
        """APLR 维护指令组（仅管理员）"""
        pass

    @aplr_maintenance_group.command("extract_history")
    async def extract_chat_history_command(self, event: AstrMessageEvent):
        """手动提取指定日期的聊天记录。用法：/APLR_maintenance extract_history [日期]（格式YYYY-MM-DD）。"""
        args = event.message_str.strip().split()
        if len(args) < 3: # /APLR_maintenance extract_history [date]
            yield event.plain_result("用法：/APLR_maintenance extract_history [YYYY-MM-DD]")
            return
        
        date_str = args[2]
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            yield event.plain_result("日期格式不正确，请使用 YYYY-MM-DD")
            return
        
        yield event.plain_result(f"正在提取 {date_str} 的聊天记录...")
        
        try:
            core_db_path = Path.cwd() / "data" / "data_v4.db"
            effective_ids = self._get_effective_user_ids(date_str)
            for target_user_id in effective_ids:
                clean_dialogue_with_different_limits(
                    db_path=core_db_path,
                    output_dir=self.dialog_folder,
                    username=self.username,
                    ai_name=self.ai_name,
                    platform="AstrBot",
                    target_date=date_str,
                    target_user_id=target_user_id,
                    day_boundary_config=self.config.get("day_boundary_config", {})
                )
            yield event.plain_result(f"✅ 已成功提取 {date_str} 的聊天记录。")
        except Exception as e:
            logger.error(f"提取聊天记录失败: {e}")
            yield event.plain_result(f"❌ 提取聊天记录失败: {e}")

    @aplr_maintenance_group.command("load_model")
    async def load_model_command(self, event: AstrMessageEvent):
        """手动或提前一键加载/下载向量模型。用法：/APLR_maintenance load_model。"""
        yield event.plain_result("⏳ 正在启动/下载并加载向量模型，如果是首次加载可能需要下载模型（耗时几分钟），请稍候...")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.vector_db._ensure_model)
            yield event.plain_result("✅ 向量模型加载并就绪成功！")
        except Exception as e:
            logger.error(f"[APLR] 手动加载向量模型失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 向量模型加载失败: {e}\n建议检查网络连接或尝试手动下载模型。\n手动下载模型方法\n1. 访问镜像站下载地址：hf-mirror.com/.../paraphrase-multilingual-MiniLM-L12-v2 \n2. （像使用github一样）下载该页面下的所有文件（尤其是 pytorch_model.bin 或 model.safetensors）。 \n3. 在本地创建一个文件夹（例如 D:\AI_Models\paraphrase-multilingual-MiniLM-L12-v2），把下载的文件全部放进去 \n4. 修改插件配置：在插件设置的 embedding_model 项中，直接填写这个本地文件夹的绝对路径（例如 D:\AI_Models\paraphrase-multilingual-MiniLM-L12-v2）。\n5. 重启 AstrBot。")

    @aplr_maintenance_group.command("delete_daily_summary")
    async def delete_daily_summary_command(self, event: AstrMessageEvent):
        """删除指定日期的所有事件、日总结及其向量与连接，并清理对应的对话记录文件。用法：/APLR_maintenance delete_daily_summary [日期]（格式YYYY-MM-DD）。"""
        args = event.message_str.strip().split()
        if len(args) < 3:
            yield event.plain_result("用法：/APLR_maintenance delete_daily_summary [YYYY-MM-DD]")
            return
            
        date_str = args[2]
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            yield event.plain_result("日期格式不正确，请使用 YYYY-MM-DD")
            return
            
        yield event.plain_result(f"⏳ 正在删除 {date_str} 的所有本地记忆记录与向量...")
        
        try:
            deleted_evs, themes_to_up, themes_to_del = self.db.delete_summary_by_date(date_str)
            if deleted_evs:
                self.vector_db.delete_events(deleted_evs)
            if themes_to_del:
                self.vector_db.delete_themes(themes_to_del)
                
            if themes_to_up:
                event_collection = self.vector_db.client.get_collection("events")
                theme_collection = self.vector_db.theme_collection
                for theme_id in themes_to_up:
                    theme_events = self.db.get_events_by_theme(theme_id)
                    if theme_events:
                        t_event_ids = [te['event_id'] for te in theme_events]
                        try:
                            res = event_collection.get(ids=t_event_ids, include=['embeddings'])
                            embeddings = res.get('embeddings')
                            if embeddings is not None and len(embeddings) > 0:
                                embs = np.array(embeddings, dtype=np.float32)
                                centroid = np.mean(embs, axis=0)
                                theme_collection.upsert(
                                    ids=[theme_id],
                                    embeddings=[centroid.tolist()]
                                )
                        except Exception as theme_vec_err:
                            logger.error(f"[APLR] 重新计算受影响主题 {theme_id} 的重心向量失败: {theme_vec_err}")
            
            deleted_files_count = 0
            if self.dialog_folder.exists() and self.dialog_folder.is_dir():
                for f in self.dialog_folder.iterdir():
                    if f.is_file() and f.name.startswith(date_str):
                        try:
                            f.unlink()
                            deleted_files_count += 1
                        except Exception as file_err:
                            logger.error(f"[APLR] 删除当日对话文件失败 {f.name}: {file_err}")
            
            msg = f"✅ 已成功清除 {date_str} 的数据：\n- 删除了 {len(deleted_evs)} 个事件及其向量\n- 删除了 {len(themes_to_del)} 个空主题"
            if themes_to_up:
                msg += f"\n- 更新了 {len(themes_to_up)} 个关联主题的重心"
            if deleted_files_count > 0:
                msg += f"\n- 删除了 {deleted_files_count} 个当日的聊天日志文件"
            else:
                msg += f"\n- 未在该日发现对应的聊天日志文件"
                
            yield event.plain_result(msg)
        except Exception as e:
            logger.error(f"[APLR] 手动删除当日数据失败 {date_str}: {e}", exc_info=True)
            yield event.plain_result(f"❌ 删除数据失败: {e}")

    @aplr_maintenance_group.command("vectorize")
    async def vectorize_events_command(self, event: AstrMessageEvent):
        """将指定日期的事件向量化并存入向量数据库。用法：/APLR_maintenance vectorize [日期|all]（格式YYYY-MM-DD）。"""
        args = event.message_str.strip().split()
        if len(args) < 3:
            yield event.plain_result("用法：/APLR_maintenance vectorize [YYYY-MM-DD] 或 /APLR_maintenance vectorize all")
            return
        
        param = args[2].lower()
        
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
                
                # 3. 分批向量化事件
                batch_size = 50
                total = len(all_events)
                for i in range(0, total, batch_size):
                    batch = all_events[i:i + batch_size]
                    self.vector_db.add_events(batch)
                    if i % 100 == 0 or i + batch_size >= total:
                        logger.info(f"[APLR] 事件向量化进度: {min(i + batch_size, total)}/{total}")
                
                # 4. 向量化主题 (现场计算重心)
                all_themes = self.db.get_all_thematic_memories()
                themes_to_add = []
                if all_themes:
                    event_collection = self.vector_db.client.get_collection("events")
                    
                    for t in all_themes:
                        theme_id = t['theme_id']
                        # 获取该主题下的所有事件 ID
                        theme_events = self.db.get_events_by_theme(theme_id)
                        if not theme_events:
                            continue
                        
                        t_event_ids = [te['event_id'] for te in theme_events]
                        # 从向量库获取 these 事件的向量
                        res = event_collection.get(ids=t_event_ids, include=['embeddings'])
                        embeddings = res.get('embeddings')
                        if embeddings is not None and len(embeddings) > 0:
                            # 计算重心
                            embs = np.array(embeddings, dtype=np.float32)
                            centroid = np.mean(embs, axis=0)
                            themes_to_add.append({
                                "theme_id": theme_id,
                                "centroid": centroid.tolist(),
                                "summary": t['summary']
                            })
                    
                    if themes_to_add:
                        self.vector_db.add_themes(themes_to_add)
                        logger.info(f"[APLR] 已重新计算并向量化 {len(themes_to_add)} 个主题。")
                
                yield event.plain_result(f"✅ 已成功重新向量化全部 {total} 个事件和 {len(themes_to_add)} 个主题。")
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

    @aplr_maintenance_group.command("update_nodes")
    async def update_nodes_command(self, event: AstrMessageEvent):
        """根据指定日期的已有事件提取并更新记忆节点。用法：/APLR_maintenance update_nodes [日期]（格式YYYY-MM-DD）。"""
        args = event.message_str.strip().split()
        if len(args) < 3:
            yield event.plain_result("用法：/APLR_maintenance update_nodes [YYYY-MM-DD]")
            return
            
        date_str = args[2]
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

            # 获取现有节点背景 (提取节点时使用优化的检索逻辑)
            dr_nodes_config = self.config.get("dailyreview_nodes", {})
            include_reflection = dr_nodes_config.get("include_reflection", False)
            
            # 注意：即便开启了 include_reflection，也不在检索背景时使用反思内容，以防干扰关键词匹配
            search_text = "\n".join([e.get('narrative', '') for e in events])
            nodes_objs = await self._get_nodes_for_summary(search_text, include_username=False)
            if nodes_objs:
                logger.debug(f"[APLR] 提取节点环节提供的参考节点: {[n['name'] for n in nodes_objs]}")
            
            existing_nodes_context = ""
            if nodes_objs:
                existing_nodes_context = "\n".join([f"- {n['name']}: {n['description']}" for n in nodes_objs])

            nodes_result = await summarizer.extract_nodes_from_events(events, date_str, existing_nodes_context=existing_nodes_context, include_reflection=include_reflection)
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

    @filter.command_group("APLR_recall")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def aplr_recall_group(self):
        """APLR 记忆检索指令组（仅管理员）"""
        pass

    @aplr_recall_group.command("memory")
    async def recall_memory_command(self, event: AstrMessageEvent):
        """根据输入文本检索最相关的记忆。用法：/APLR_recall memory [文本内容] [条数(可选)]。人类专用版！（仅管理员）"""
        msg = event.message_str.strip()
        # 移除可能的斜杠和命令名部分
        content = re.sub(r'^/?APLR_recall\s+memory\s*', '', msg).strip()
        
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

    @llm_tool(name="deep_recall_tool")
    async def deep_recall_tool(self, event: AstrMessageEvent, target: str, mode: str = "") -> str:
        """深度回想工具。当你已经获得一个特定的线索（ID或日期）并希望挖掘更多细节时使用。
        
        支持以下输入格式：
        1. 事件ID (以 evt_ 开头)：回想特定往事的深度感想和细节。
        2. 主题ID (以 theme_ 开头)：从宏观主题中联想出 3 个具体的代表性记忆片段。
        3. 日期 (YYYY-MM-DD 格式)：回想那一天的整体心境和感悟。
        
        Args:
            target(string): 要深度回想的目标（ID或日期）。
            mode(string): 联想模式（仅对主题有效）。可选：'类人'(均衡)、'时间'(侧重近期)、'情绪'(侧重强烈情感)、'随机'(完全随机)。留空则使用系统默认权重。
        """
        if not target:
            return "错误：需要提供回想目标 target。"
        
        target = target.strip()
        
        # 1. 处理事件 ID
        if target.startswith("evt_"):
            return await self._get_event_reflection_logic(target)
            
        # 2. 处理主题 ID
        elif target.startswith("theme_"):
            return await self._get_theme_recall_logic(target, mode)
            
        # 3. 处理日期格式 (YYYY-MM-DD)
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', target):
            return await self._get_daily_reflection_logic(target)
            
        else:
            return f"错误：无法识别的回想目标格式 '{target}'。请确保输入的是事件ID(evt_...)、主题ID(theme_...)或日期(YYYY-MM-DD)。"

    async def _get_theme_recall_logic(self, theme_id: str, mode: str = "") -> str:
        """从主题中进行概率不均等的随机抽取事件"""
        theme = self.db.get_thematic_memory(theme_id)
        if not theme:
            return f"未找到 ID 为 {theme_id} 的主题记忆。"
            
        events = self.db.get_events_by_theme(theme_id)
        if not events:
            return f"主题 {theme_id} 下暂时没有关联的具体事件。"
            
        # 确定权重模式
        mode = mode.strip()
        if mode == "类人":
            w_time, w_importance, w_intensity = 1.0, 1.0, 1.0
        elif mode == "时间":
            w_time, w_importance, w_intensity = 1.0, 0.0, 0.0
        elif mode == "情绪":
            w_time, w_importance, w_intensity = 0.0, 0.0, 1.0
        elif mode == "随机":
            w_time, w_importance, w_intensity = 0.0, 0.0, 0.0
        else:
            # 默认使用配置
            w_time = self.config.get("weight_time", 1.0)
            w_importance = self.config.get("weight_importance", 1.0)
            w_intensity = self.config.get("weight_emotional_intensity", 1.0)
            if mode:
                logger.debug(f"[APLR] 无效的回想模式 '{mode}'，已回退到系统默认权重。")

        # 计算每个事件的权重
        now = datetime.now().date()
        weights = []
        for ev in events:
            try:
                ev_date = datetime.strptime(ev['date'], "%Y-%m-%d").date()
                days_diff = (now - ev_date).days
                date_weight = 1.0 / (1 + math.log1p(days_diff / 30.0))
            except:
                date_weight = 0.1
                
            importance = ev.get('importance', 5)
            intensity = ev.get('emotional_intensity', 5)
            
            # 综合权重公式
            score = (importance ** w_importance) * \
                    (intensity ** w_intensity) * \
                    (date_weight ** w_time)
            weights.append(max(0.001, score)) # 确保权重不为0
            
        # 概率不均等随机抽取 (固定为 3 条)
        import random
        num_to_sample = min(len(events), 3)
        
        sampled_events = []
        available_indices = list(range(len(events)))
        available_weights = list(weights)
        
        for _ in range(num_to_sample):
            if not available_indices: break
            idx_in_available = random.choices(range(len(available_indices)), weights=available_weights, k=1)[0]
            real_idx = available_indices.pop(idx_in_available)
            available_weights.pop(idx_in_available)
            sampled_events.append(events[real_idx])
            
        # 格式化输出
        resp = f"📂 **【主题深度回想：{theme_id}】** (模式: {mode if mode in ['类人', '时间', '情绪', '随机'] else '默认'})\n"
        resp += f"💭 核心感悟: {theme['summary']}\n\n"
        resp += f"💡 联想到了以下 {len(sampled_events)} 个代表性片段：\n"
        
        # 按日期排序展示
        sampled_events.sort(key=lambda x: x['date'])
        for ev in sampled_events:
            resp += f"  - [{ev['date']}] {ev['narrative']} [ID: {ev['event_id']}]\n"
            
        if self.config.get("encourage_deep_recall", False):
            resp += "\n（注：若想进一步了解某个片段的细节，可继续对该 事件ID 使用 `deep_recall_tool`）"
            
        return resp

    async def _get_event_reflection_logic(self, event_id: str) -> str:
        ev = self.db.get_event_by_id(event_id)
        if ev:
            reflection = ev.get('reflection', '')
            if reflection:
                res = f"关于事件 {event_id} 的深度观察和感想：\n{reflection}"
            else:
                res = f"该事件 ({event_id}) 暂时没有记录深度感想呢。"
            
            if self.config.get("encourage_deep_recall", False):
                res += f"\n（注：若希望回忆该事件发生当天 ({ev['date']}) 的整体心境，可选择对日期使用 `deep_recall_tool`。）"
            return res
        return f"未找到 ID 为 {event_id} 的事件。"

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

    @aplr_recall_group.command("deep")
    async def deep_recall_command(self, event: AstrMessageEvent):
        """深度回想。用法：/APLR_recall deep [ID或日期] [模式(可选)]。"""
        msg = event.message_str.strip()
        # 移除命令名部分
        content = re.sub(r'^/?APLR_recall\s+deep\s*', '', msg).strip()
        
        if not content:
            yield event.plain_result("用法：/APLR_recall deep [事件ID|主题ID|日期] [模式(可选)]\n模式支持：类人、时间、情绪、随机")
            return
            
        parts = content.split()
        target = parts[0]
        mode = ""
        if len(parts) > 1:
            mode = parts[1]
            
        res = await self.deep_recall_tool(event, target, mode)
        yield event.plain_result(res)

    @aplr_recall_group.command("recent")
    async def recall_recent_events_command(self, event: AstrMessageEvent, days: str = "7", min_score: str = "20"):
        """获取最近一段时间内比较重要的、或情感强烈的事件。用法：/APLR_recall recent [天数] [筛选分数]"""
        try:
            d = int(days)
            s = int(min_score)
        except ValueError:
            yield event.plain_result("请输入有效的数字（如：/APLR_recall recent 7 20）。")
            return
            
        res = await self.recall_recent_events_tool(event, d, s)
        yield event.plain_result(res)

    @aplr_recall_group.command("node")
    async def recall_node_command(self, event: AstrMessageEvent):
        """搜索某个记忆节点（实体或概念）。用法：/APLR_recall node [名称]"""
        msg = event.message_str.strip()
        # 移除可能的斜杠和命令名部分
        query = re.sub(r'^/?APLR_recall\s+node\s*', '', msg).strip()
        
        if not query:
            yield event.plain_result("请输入要搜索的节点名称。")
            return
        
        nodes = self.db.search_nodes(query, limit=2)
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

    @aplr_maintenance_group.command("write_node")
    async def write_node_command(self, event: AstrMessageEvent):
        """手动写入或更新记忆节点。用法：/APLR_maintenance write_node [节点名] [类型] [描述]"""
        msg = event.message_str.strip()
        # 移除命令名部分
        content = re.sub(r'^/?APLR_maintenance\s+write_node\s*', '', msg).strip()
        
        if not content:
            yield event.plain_result("用法：/APLR_maintenance write_node [节点名] [类型] [描述]")
            return
            
        parts = content.split(maxsplit=2)
        if len(parts) < 3:
            yield event.plain_result("参数不足。用法：/APLR_maintenance write_node [节点名] [类型] [描述]\n示例：/APLR_maintenance write_node olozhika 人物 Lanya的好朋友")
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
        0. 记忆节点的用途是存长期身份/状态/规则，避免一次性琐碎细节。
        1. 请务必先使用 `recall_node_tool` 查询相应节点是否存在。
        2. 如果节点已存在，请务必在保留完整核心事实（如身份、关键背景）的基础上，根据今日信息更新其状态或追加新进展。记忆节点中已有的信息是你宝贵的记忆财富，不要轻易丢弃其中的内容，除非里面的内容已经不合时宜。
        
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
        nodes = self.db.search_nodes(name, limit=2)
        if nodes:
            res = f"找到以下关于 {name} 的相关 node 信息：\n"
            for node in nodes:
                last_updated_date = node['last_updated'].split(' ')[0] if node['last_updated'] else "未知"
                res += f"- {node['name']} ({node['type']}): {node['description']} (最后更新于 {last_updated_date})\n"
            return res
        return f"未找到关于 {name} 的节点信息。"

    async def _get_memory_retrieval_text(self, query: str, count: int = None) -> str:
        # 1. 获取基础配置
        if count:
            m1 = count
            m2 = 0 # 既然指定了数量，通常是为了精确检索，减少随机干扰
            top_n = max(10, count * 2)
        else:
            top_n = self.config.get("top_n_events", 10)
            m1 = self.config.get("m1_top_events", 3)
            m2 = self.config.get("m2_random_events", 2)

        # 2. 向量检索 (同时检索事件和主题)
        search_query = query
        if self.ai_name:
            pattern = r'(?<!自)我(?![们])'
            search_query = re.sub(pattern, self.ai_name, query)
            
        search_results = self.vector_db.search_all(search_query, top_n_events=top_n, top_n_themes=5)
        if not search_results:
            return "暂时没有找到相关的记忆呢。"

        # 3. 统一评分竞技场
        candidates = []
        now = datetime.now().date()
        
        # 预处理关键词权重用于事件评分
        query_terms = set([word for word in jieba.lcut(search_query) if len(word) > 1])
        
        for res in search_results:
            rid = res['id']
            relevance = res['relevance']
            rtype = res['type']
            
            if rtype == "event":
                ev = self.db.get_event_by_id(rid)
                if ev:
                    # 计算事件综合得分 (参考 _rank_events 逻辑)
                    score = self._calculate_single_event_score(ev, relevance, query_terms, now)
                    candidates.append({
                        "type": "event",
                        "score": score,
                        "data": ev
                    })
            elif rtype == "theme":
                theme = self.db.get_thematic_memory(rid)
                if theme:
                    # 获取该主题下所有事件，用于计算时间权重和平均重要性
                    theme_events = self.db.get_events_by_theme(rid)
                    theme_score = self._calculate_theme_score(theme, theme_events, relevance, now)
                    candidates.append({
                        "type": "theme",
                        "score": theme_score,
                        "data": theme
                    })

        if not candidates:
            return "检索到了 ID 但没能从数据库找到详细信息。"

        # 4. 统一排序
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 5. 筛选 M1 + M2 (平等竞争)
        selected_items = candidates[:m1]
        remaining = candidates[m1:]
        if remaining and m2 > 0:
            selected_items.extend(random.sample(remaining, min(len(remaining), m2)))

        # 6. 格式化输出 (基于主题的分组展示)
        resp = f"🔍 回想起了相关记忆：\n\n"
        
        # 组织数据结构：{ theme_id: { "theme": theme_data, "events": [event_data] } }
        # 使用 "none" 作为无主题事件的 key
        groups = {}
        group_order = [] # 保持首次出现的顺序
        
        for item in selected_items:
            if item['type'] == "theme":
                t = item['data']
                tid = t['theme_id']
                if tid not in groups:
                    groups[tid] = {"theme": t, "events": []}
                    group_order.append(tid)
                else:
                    groups[tid]["theme"] = t
            else:
                ev = item['data']
                theme_info = self.db.get_theme_by_event_id(ev['event_id'])
                if theme_info:
                    tid = theme_info['theme_id']
                    if tid not in groups:
                        groups[tid] = {"theme": theme_info, "events": [ev]}
                        group_order.append(tid)
                    else:
                        groups[tid]["events"].append(ev)
                else:
                    if "none" not in groups:
                        groups["none"] = {"theme": None, "events": [ev]}
                        group_order.append("none")
                    else:
                        groups["none"]["events"].append(ev)

        # 按顺序渲染分组
        for i, tid in enumerate(group_order):
            group = groups[tid]
            
            # 1. 先渲染事件部分 (具体往事)
            for ev in group["events"]:
                # 强化记忆
                self.db.reinforce_memory([ev['event_id']])
                resp += f"📜 **【具体往事】** ({ev['date']})\n"
                resp += f"   - {ev['narrative']} [ID: {ev['event_id']}]\n"
            
            # 2. 再渲染主题部分 (宏观感悟)
            if group["theme"]:
                t = group["theme"]
                # 如果有事件，加一个小标题提示这是这些事件的总结
                prefix = "💡 以上事件所属的主题感悟：" if group["events"] else ""
                resp += f"📂 **【主题记忆】{t['theme_id']}** {prefix}\n"
                resp += f"   💭 核心感悟: {t['summary']}\n"
            
            # 分隔符 (如果不是最后一组)
            if i < len(group_order) - 1:
                resp += "\n" + "—" * 20 + "\n\n"

        if self.config.get("encourage_deep_recall", False):
            resp += "\n（注：若想进一步回忆某个事件的细节，可以使用 `deep_recall_tool` 传入事件ID）"
            
        return resp

    def _calculate_theme_score(self, theme, theme_events, vector_relevance, now_date) -> float:
        """计算主题记忆的综合得分"""
        if not theme_events:
            return (vector_relevance / 100.0) ** 4 * 5.0 # 兜底
            
        # 1. 计算平均重要性和情感强度
        avg_importance = sum(e.get('importance', 5) for e in theme_events) / len(theme_events)
        avg_intensity = sum(e.get('emotional_intensity', 5) for e in theme_events) / len(theme_events)
        
        # 2. 计算代表性日期权重 (取第 0.75 新的事件时间)
        try:
            sorted_dates = sorted([e['date'] for e in theme_events])
            n = len(sorted_dates)
            # 取第 0.75 位置的日期作为代表（介于中位数和最新值之间）
            target_idx = int(0.75 * (n - 1))
            representative_date_str = sorted_dates[target_idx]
            representative_date = datetime.strptime(representative_date_str, "%Y-%m-%d").date()
            days_diff = (now_date - representative_date).days
            date_weight = 1.0 / (1 + math.log1p(days_diff / 30.0))
        except:
            date_weight = 0.5
            
        # 3. 获取配置权重
        w_time = self.config.get("weight_time", 1.0)
        w_importance = self.config.get("weight_importance", 1.0)
        w_intensity = self.config.get("weight_emotional_intensity", 1.0)
        
        relevance_score = (vector_relevance / 100.0) ** 4
        
        # 4. 综合公式
        # 这里的 0.8 是调节因子，确保主题不会因为聚合效应而过度压制单个极其重要的孤立事件
        score = (avg_importance ** w_importance) * \
                (avg_intensity ** w_intensity) * \
                (date_weight ** w_time) * \
                relevance_score * 0.8
                
        return score

    def _calculate_single_event_score(self, ev, vector_relevance, query_terms, now_date) -> float:
        """计算单个事件的综合记忆得分"""
        try:
            ev_date = datetime.strptime(ev['date'], "%Y-%m-%d").date()
            days_diff = (now_date - ev_date).days
            date_weight = 1.0 / (1 + math.log1p(days_diff / 30.0))
        except:
            date_weight = 0.1
        
        importance = ev.get('importance', 1)
        intensity = ev.get('emotional_intensity', 1)
        relevance_score = (vector_relevance / 100.0) ** 4
        
        # 关键词加成
        keyword_score = 0.0
        narrative = ev.get('narrative', '')
        for term in query_terms:
            if term in narrative:
                keyword_score += 1.0
        keyword_bonus = 1.0 + (keyword_score * 0.5)
        
        # 强化计数加成
        re_intensity = self.config.get("reinforcement_intensity", 1.0)
        reinforcement_boost = ev.get('reinforcement_count', 0) * 0.1 * re_intensity
        
        # 综合公式 (参考 _rank_events)
        w_importance = self.config.get("weight_importance", 1.0)
        w_intensity = self.config.get("weight_emotional_intensity", 1.0)
        
        score = (importance ** w_importance) * \
                (intensity ** w_intensity) * \
                relevance_score * \
                date_weight * \
                keyword_bonus + \
                reinforcement_boost
        return score

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
                theme_str = ""
                if ev.get('theme_info'):
                    t = ev['theme_info']
                    theme_str = f" [所属主题: {t['theme_id']} - {t['summary'][:50]}...]"
                narrative += f"  - {ev['narrative']} [ID: {ev['event_id']}]{theme_str}\n"
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
        """进行每日总结用的工具，回顾并总结今日或指定日期的交流。只在执行关于每日总结的Cron job，或监护人要求你使用时触发，此外的任何情况请不要主动触发！
        
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

    async def _daily_summary_logic(self, event: AstrMessageEvent = None, date_str: str = None):
        # 如果没有直接传入 date_str，则尝试从消息中解析
        if not date_str:
            args = event.message_str.strip().split() if event else []
            # 兼容处理：跳过命令名或斜杠命令名
            if args and (args[0] in ["daily_summary", "/daily_summary", "daily_summary_command", "/daily_summary_command"]):
                args = args[1:]
                
            if args:
                date_str = args[0]
            else:
                date_str = self._get_logical_date(datetime.now())
        
        # 再次确保 date_str 是有效的字符串（处理 AI 可能传回的空字符串）
        if not date_str or not date_str.strip():
            date_str = self._get_logical_date(datetime.now())

        effective_user_ids = self._get_effective_user_ids(date_str)

        # 检查是否已经存在该日期的实时记录 JSON
        folder = self.dialog_folder
        json_pattern = f"{date_str}_dialog_*.json"
        has_local_json = any(folder.glob(json_pattern))

        # 如果没有本地记录文件，则尝试从核心数据库提取（保底逻辑）
        if not has_local_json:
            try:
                # 核心数据库通常位于 data/data_v4.db
                core_db_path = Path.cwd() / "data" / "data_v4.db"
                if core_db_path.exists():
                    logger.info(f"[APLR] 未找到日期 {date_str} 的记录文件，正在尝试从核心数据库提取...")
                    for target_user_id in effective_user_ids:
                        clean_dialogue_with_different_limits(
                            db_path=core_db_path,
                            output_dir=self.dialog_folder,
                            username=self.username,
                            ai_name=self.ai_name,
                            platform="AstrBot",
                            target_date=date_str,
                            target_user_id=target_user_id,
                            day_boundary_config=self.config.get("day_boundary_config", {})
                        )
            except Exception as e:
                logger.error(f"保底提取聊天记录失败: {e}")

        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            if event:
                yield event.plain_result("日期格式好像不对哦，记得用 YYYY-MM-DD 这种格式~")
            else:
                logger.error(f"[APLR] 自动重整回忆任务日期格式不合法: {date_str}")
            return
        
        # 寻找聊天记录文件并合并
        conversation_chunks = []
        found_any = False
        
        max_kb = self.config.get("max_dialogue_kb_per_summary", 40)
        gap_hours = self.config.get("chunk_time_gap_hours", 1.0)
 
        current_chunk = []
        current_size = 0.0
        last_time = None
 
        for target_user_id in effective_user_ids:
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
                        target_user_id=target_user_id,
                        day_boundary_config=self.config.get("day_boundary_config", {})
                    )
                except Exception as e:
                    logger.error(f"提取聊天记录失败: {e}")
                
                # 提取后再检查一次
                if not dialog_file.exists():
                    #logger.info(f"[APLR] 未读取 {date_str} ({target_user_id}) 的聊天记录，路径: {dialog_file}")
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

                    # 添加包含详细场景元数据的对话提示头，精准指导AI
                    meta = data.get("metadata", {})
                    g_name = meta.get("group_name")
                    n_name = meta.get("nickname")
                    c_type = meta.get("chat_type")
                    
                    # Session ID 判定做双重加固
                    is_group_by_session = "GroupMessage" in (target_user_id or "")
                    is_friend_by_session = "FriendMessage" in (target_user_id or "")
                    specific_id = target_user_id.split(":")[-1] if target_user_id and ":" in target_user_id else (target_user_id or "")

                    if g_name:
                        info_str = f"群聊：{g_name}"
                    elif n_name:
                        info_str = f"与 {n_name} 的私聊"
                    elif c_type == "group" or is_group_by_session:
                        info_str = f"群聊 (ID: {specific_id})"
                    elif c_type == "private" or is_friend_by_session:
                        info_str = f"私聊 (ID: {specific_id})"
                    else:
                        info_str = f"私聊：{target_user_id}"

                    header = f"\n=== 对话场景: [{info_str}] (会话ID: {target_user_id}) ===\n"
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
                            current_chunk = [f"\n=== 对话场景: [{info_str}] (会话ID: {target_user_id}) (续) ===\n"]
                            current_size = len(current_chunk[0].encode('utf-8')) / 1024.0
                        
                        current_chunk.append(msg_text)
                        current_size += msg_size
                        last_time = timestamp
            except Exception as e:
                logger.info(f"[APLR] 读取记录 ({target_user_id}) 的时候出了点小状况：{e}")

        if current_chunk:
            conversation_chunks.append("".join(current_chunk))

        # 自动发现并注入额外的上下文文件 (*_context_*.txt)
        extra_contexts = []
        for context_file in self.dialog_folder.glob(f"{date_str}_context_*.txt"):
            try:
                with open(context_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # 提取 source_name，例如 2024-01-01_context_Health.txt -> Health
                        source_name = context_file.stem.split("_context_")[-1]
                        extra_contexts.append(f"\n=== 外部上下文 ({source_name}) ===\n{content}\n")
            except Exception as e:
                logger.error(f"[APLR] 读取外部上下文文件 {context_file} 失败: {e}")
        
        if extra_contexts:
            found_any = True
            context_str = "".join(extra_contexts)
            if conversation_chunks:
                # 注入到第一段总结背景中
                conversation_chunks[0] = context_str + conversation_chunks[0]
            else:
                conversation_chunks.append(context_str)

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
                    #logger.debug(f"base sys prompt: {base_system_prompt}")
            except Exception as pe:
                logger.warning(f"[APLR] 尝试获取 Astrbot 人格设定失败: {pe}")

            # 4. 初始化总结器
            prompts_config = self.config.get("prompts", {})
            summarizer = DailySummarizer(
                llm_generate_func=llm_generate_func,
                ai_name=self.ai_name,
                base_system_prompt=base_system_prompt,
                base_user_prompt=base_user_prompt,
                search_events_func=self._search_events_for_summary,
                prompt_event_summary=prompts_config.get("event_summary", ""),
                prompt_memory_node=prompts_config.get("memory_node", "")
            )

            # 获取现有节点背景 (总结时使用优化的检索逻辑)
            full_text = "\n".join(conversation_chunks)
            nodes = await self._get_nodes_for_summary(full_text, include_username=False)
            if nodes:
                logger.debug(f"[APLR] 每日总结环节提供的参考节点: {[n['name'] for n in nodes]}")

            existing_nodes_context = ""
            if nodes:
                existing_nodes_context = "\n".join([f"- {n['name']}: {n['description']}" for n in nodes])

            # 5. 异步调用总结生成
            dr_nodes_config = self.config.get("dailyreview_nodes", {})
            summary = await summarizer.generate_summary(
                conversation_chunks, 
                date_str, 
                existing_nodes_context=existing_nodes_context,
                include_reflection=dr_nodes_config.get("include_reflection", False)
            )

            # --- 记忆固化：增量更新 ---
            try:
                if summary and summary.events:
                    logger.info("[APLR] 正在执行增量记忆固化...")
                    if not self.consolidator:
                        prompts_config = self.config.get("prompts", {})
                        self.consolidator = ThematicConsolidator(
                            db=self.db,
                            vector_db=self.vector_db,
                            llm_generate_func=llm_generate_func,
                            ai_name=self.ai_name,
                            persona=base_system_prompt,
                            prompt_theme_summary=prompts_config.get("theme_summary", "")
                        )
                    else:
                        # 更新可能变化的 persona 和 llm 函数
                        prompts_config = self.config.get("prompts", {})
                        self.consolidator.llm_generate = llm_generate_func
                        self.consolidator.persona = base_system_prompt
                        self.consolidator.prompt_theme_summary = prompts_config.get("theme_summary", "")
                    
                    await self.consolidator.incremental_consolidation()
            except Exception as ce:
                logger.error(f"[APLR] 增量固化失败: {ce}")

        except Exception as e:
            logger.error(f"[APLR] 生成总结过程中发生错误: {e}", exc_info=True)
            logger.info(f"[APLR] 生成总结过程中发生错误：{e}")
            return

        if summary is None:
            logger.info("[APLR] 诶，没能整理出有意义的总结呢")
            return

        try:
            # 自动清除旧的当日事件、日总结及其向量与连接，避免出现重复
            try:
                logger.info(f"[APLR] 开始检查并自动清除 {date_str} 的旧事件与日总结记录...")
                deleted_evs, themes_to_up, themes_to_del = self.db.delete_summary_by_date(date_str)
                if deleted_evs:
                    self.vector_db.delete_events(deleted_evs)
                    logger.info(f"[APLR] 已自动清除 {len(deleted_evs)} 条已存在的当日事件向量")
                if themes_to_del:
                    self.vector_db.delete_themes(themes_to_del)
                    logger.info(f"[APLR] 已自动清除 {len(themes_to_del)} 个空主题向量")
                if themes_to_up:
                    event_collection = self.vector_db.client.get_collection("events")
                    theme_collection = self.vector_db.theme_collection
                    for theme_id in themes_to_up:
                        theme_events = self.db.get_events_by_theme(theme_id)
                        if theme_events:
                            t_event_ids = [te['event_id'] for te in theme_events]
                            try:
                                res = event_collection.get(ids=t_event_ids, include=['embeddings'])
                                embeddings = res.get('embeddings')
                                if embeddings is not None and len(embeddings) > 0:
                                    embs = np.array(embeddings, dtype=np.float32)
                                    centroid = np.mean(embs, axis=0)
                                    theme_collection.upsert(
                                        ids=[theme_id],
                                        embeddings=[centroid.tolist()]
                                    )
                            except Exception as theme_vec_err:
                                logger.error(f"[APLR] 重新计算受影响主题 {theme_id} 的重心向量失败: {theme_vec_err}")
                    logger.info(f"[APLR] 已同步更新受影响的 {len(themes_to_up)} 个主题的重心向量")
            except Exception as clear_err:
                logger.error(f"[APLR] 自动清理 {date_str} 历史事件和总结失败: {clear_err}", exc_info=True)

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
            if event:
                yield event.plain_result(f"✨ {date_str} 的回忆整理好啦！\n\n")
                resp = f"💭 我的感悟：{summary.daily_reflection}\n"
                resp += f"📍 我记下了 {len(summary.events)} 个印象深刻的瞬间。"
                yield event.plain_result(resp)
            else:
                logger.info(f"[APLR] ✨ {date_str} 的回忆自动整理好了！")
                logger.info(f"[APLR] 我的感悟：{summary.daily_reflection}")
                logger.info(f"[APLR] 我记下了 {len(summary.events)} 个印象深刻的瞬间。")
        except Exception as e:
            logger.exception("存入数据库失败")
            if event:
                yield event.plain_result(f"[APLR] 虽然想起来了，但没能存进记忆库：{e}")
            else:
                logger.error(f"[APLR] 自动重整回忆任务存入数据库失败: {e}")

    @filter.command("memory_consolidation")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def memory_consolidation_command(self, event: AstrMessageEvent):
        """执行全局记忆主题归类（仅管理员），重新对所有记忆进行聚类和总结。建议记忆数据库有200条以上事件时使用，每数月到数年重新执行一次本函数。用法：/memory_consolidation"""
        yield event.plain_result("🚀 正在启动全局记忆主题归类（大固化），这可能需要较长时间，请耐心等待...")
        
        try:
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

            base_system_prompt = self.config.get("default_prompt", "")
            try:
                persona = await self.context.persona_manager.get_default_persona_v3(umo=umo)
                if persona and persona.get('prompt'):
                    base_system_prompt = (persona['prompt'] + "\n" + base_system_prompt) if base_system_prompt else (persona['prompt'] + "\n")
            except:
                pass

            if not self.consolidator:
                prompts_config = self.config.get("prompts", {})
                self.consolidator = ThematicConsolidator(
                    db=self.db,
                    vector_db=self.vector_db,
                    llm_generate_func=llm_generate_func,
                    ai_name=self.ai_name,
                    persona=base_system_prompt,
                    prompt_theme_summary=prompts_config.get("theme_summary", "")
                )
            else:
                prompts_config = self.config.get("prompts", {})
                self.consolidator.llm_generate = llm_generate_func
                self.consolidator.persona = base_system_prompt
                self.consolidator.prompt_theme_summary = prompts_config.get("theme_summary", "")

            await self.consolidator.big_consolidation()
            yield event.plain_result("✅ 全局记忆固化完成！所有记忆已重新聚类并生成总结。")
        except Exception as e:
            logger.error(f"大固化失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 大固化失败: {e}")

    @aplr_recall_group.command("theme")
    async def recall_theme_command(self, event: AstrMessageEvent):
        """查看已固化的主题记忆。用法：/APLR_recall theme [主题ID(可选)]"""
        args = event.message_str.strip().split()
        if len(args) > 2: # /APLR_recall theme [id]
            theme_id = args[2]
            events = self.db.get_events_by_theme(theme_id)
            if not events:
                yield event.plain_result(f"未找到主题 {theme_id}")
                return
            
            # 获取主题总结
            themes = self.db.get_all_thematic_memories()
            summary = next((t['summary'] for t in themes if t['theme_id'] == theme_id), "暂无总结")
            
            resp = f"📂 **主题: {theme_id}**\n"
            resp += f"💭 **我的感悟:**\n{summary}\n\n"
            resp += f"📍 **包含事件 ({len(events)} 条):**\n"
            for e in events[:20]: # 最多显示20条
                resp += f"  - [{e['date']}] {e['narrative']}\n"
            if len(events) > 20:
                resp += f"  ... 以及其他 {len(events)-20} 条记录。"
            yield event.plain_result(resp)
        else:
            themes = self.db.get_all_thematic_memories()
            if not themes:
                yield event.plain_result("目前还没有固化的主题记忆，请先执行 /big_consolidation")
                return
            
            resp = f"📂 **已固化的主题记忆 ({len(themes)} 个):**\n\n"
            for t in themes:
                resp += f"🔹 **{t['theme_id']}** ({t['event_count']} 条事件)\n"
                resp += f"   摘要: {t['summary'][:50]}...\n"
            resp += "\n使用 `/recall_theme_command [主题ID]` 查看详情。"
            yield event.plain_result(resp)

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
