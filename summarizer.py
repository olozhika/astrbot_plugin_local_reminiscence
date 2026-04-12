import json
import re
from typing import List
from astrbot.api import logger
from .models import DailySummary

class DailySummarizer:
    def __init__(self, llm_generate_func, ai_name: str = "Lanya", base_system_prompt: str = "", base_user_prompt: str = "", search_events_func=None, prompt_event_summary: str = "", prompt_memory_node: str = ""):
        self.llm_generate = llm_generate_func
        self.ai_name = ai_name
        self.base_system_prompt = base_system_prompt.strip()
        self.base_user_prompt = base_user_prompt.strip()
        self.search_events = search_events_func
        self.prompt_event_summary = prompt_event_summary
        self.prompt_memory_node = prompt_memory_node

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON 结构体，处理 Markdown 代码块和前后无关文字"""
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if match:
            content = match.group(1).strip()
        else:
            obj_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
            if obj_match:
                content = obj_match.group(1).strip()
            else:
                content = text.strip()
        
        # 尝试初步解析，如果失败则尝试修复
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            return self._repair_json(content)

    def _repair_json(self, s: str) -> str:
        """启发式修复 LLM 生成的破损 JSON（如未转义的双引号、多余的逗号）"""
        # 1. 处理末尾多余的逗号
        s = re.sub(r',\s*([\]}])', r'\1', s)
        
        # 2. 处理字符串内部未转义的双引号
        # 逻辑：寻找 ": "..." 结构，并对中间内容进行转义
        # 改进：通过正向预查确保只有在遇到真正的字段结束（", 后面跟 " 或 }/]）时才停止匹配
        def escape_internal(match):
            start, content, end = match.groups()
            # 将中间内容中所有未转义的 " 转义
            fixed_content = re.sub(r'(?<!\\)"', r'\"', content)
            return start + fixed_content + end

        # 匹配模式：冒号 + 引号 + 内容 + (引号 + 真正的结构分隔符)
        s = re.sub(r'(\s*:\s*")([\s\S]*?)("\s*(?:,(?=\s*")|\s*[}\]]))', escape_internal, s)
        
        return s

    async def extract_nodes_from_events(self, events: List[dict], date_str: str, existing_nodes_context: str = "") -> tuple[list, list] | None:
        """从已有的事件叙述中提取记忆节点"""
        if not events:
            return [], []
            
        events_text = "\n---\n".join([f"事件ID: {ev['event_id']}\n叙述: {ev['narrative']}" for ev in events])
        
        from .models import MemoryNode
        schema_dict = {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": MemoryNode.model_json_schema()
                },
                "deleted_nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "需要删除或合并的冗余节点名称列表"
                }
            },
            "required": ["nodes", "deleted_nodes"]
        }
        schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2).replace("{ai_name}", self.ai_name)
        
        context_text = f"\n【已知记忆节点背景】\n{existing_nodes_context}\n" if existing_nodes_context else ""
        
        system_prompt = f'''你正在从一组你之前总结的事件中提取记忆节点。必须严格按照以下 JSON Schema 输出 JSON 数据：
{schema_str}

【提取规则】
{self.prompt_memory_node}
    - 今天的日期是 {date_str}。
{context_text}
'''
        if self.base_system_prompt:
            system_prompt = self.base_system_prompt + "\n\n" + system_prompt
        try:
            llm_resp = await self.llm_generate(
                prompt=f"请从以下事件中提取记忆节点：\n\n{events_text}",
                system_prompt=system_prompt,
            )
            content = self._extract_json(llm_resp.completion_text)
            data = json.loads(content)
            nodes_data = data.get("nodes", [])
            deleted_nodes = data.get("deleted_nodes", [])
            return [MemoryNode(**n) for n in nodes_data], deleted_nodes
        except Exception as e:
            logger.error(f"提取节点时发生错误: {e}", exc_info=True)
            return None

    async def generate_summary(self, conversation_chunks: List[str], date_str: str, existing_nodes_context: str = "") -> DailySummary | None:
        """两阶段总结法：
        1. 提取今日事件与日感想（支持分段处理）。
        2. 提取/更新记忆节点。
        """
        
        # --- 第一阶段：核心总结（事件、日感想） ---
        from .models import Event
        schema_dict = {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": Event.model_json_schema()
                },
                "daily_reflection": {
                    "type": "string",
                    "description": "{ai_name}的自由总结/心得"
                }
            },
            "required": ["events", "daily_reflection"]
        }
        
        schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2).replace("{ai_name}", self.ai_name)
        
        pass1_system_prompt = f'''你正在回顾并总结今天与他人的交流，你的总结将构成你对今天的长期记忆。必须严格按照以下 JSON Schema 输出 JSON 数据：
{schema_str}

【总结规则】
{self.prompt_event_summary}
- 今天的日期是 {date_str}。
'''
        
        full_system_prompt_template = (self.base_system_prompt + "\n\n" + pass1_system_prompt) if self.base_system_prompt else pass1_system_prompt
        
        all_events = []
        all_reflections = []
        current_event_count = 0

        for i, chunk in enumerate(conversation_chunks):
            logger.info(f"[APLR] 执行第一阶段总结 (分段 {i+1}/{len(conversation_chunks)})：提取事件与日感想...")
            
            # 动态调整系统提示词，告知起始序号
            start_index = current_event_count + 1
            chunk_system_prompt = full_system_prompt_template + f"\n- **重要：请从序号 evt_{date_str.replace('-', '')}_{start_index:03d} 开始为本段事件编号 。**"
            
            user_prompt = (self.base_user_prompt + "\n\n" + chunk) if self.base_user_prompt else chunk
            
            try:
                llm_resp1 = await self.llm_generate(prompt=user_prompt, system_prompt=chunk_system_prompt)
                content1 = self._extract_json(llm_resp1.completion_text)
                data1 = json.loads(content1)
                
                chunk_events = data1.get("events", [])
                chunk_reflection = data1.get("daily_reflection", "")
                
                all_events.extend(chunk_events)
                current_event_count += len(chunk_events)
                if chunk_reflection:
                    all_reflections.append(chunk_reflection)
            except Exception as e:
                logger.error(f"分段 {i+1} 总结失败: {e}")

        if not all_events and not all_reflections:
            return None

        # --- 整合心得 (如果分了多段) ---
        final_reflection = ""
        if len(all_reflections) == 1:
            final_reflection = all_reflections[0]
        elif len(all_reflections) > 1:
            logger.info(f"[APLR] 正在整合 {len(all_reflections)} 段心得为当日心得...")
            reflections_text = "\n\n".join([f"--- 分段心得 {j+1} ---\n{r}" for j, r in enumerate(all_reflections)])
            
            consolidation_system_prompt = f"请将以下几段关于同一天 ({date_str}) 的对话心得，整合为一段连贯、精炼的心得。"
            if self.base_system_prompt:
                consolidation_system_prompt = self.base_system_prompt + "\n\n" + consolidation_system_prompt
            
            try:
                llm_resp_ref = await self.llm_generate(
                    prompt=f"请整合以下心得：\n\n{reflections_text}",
                    system_prompt=consolidation_system_prompt
                )
                final_reflection = llm_resp_ref.completion_text.strip()
            except Exception as e:
                logger.error(f"整合总心得失败: {e}")
                final_reflection = "\n\n".join(all_reflections) # 降级方案

        summary = DailySummary(
            date=date_str,
            events=all_events,
            daily_reflection=final_reflection,
            nodes=[],
            relations=[],
            deleted_nodes=[]
        )

        if summary.date != date_str:
            summary.date = date_str

        try:
            # --- 第二阶段：提取记忆节点 ---
            if summary.events:
                logger.info(f"[APLR] 执行第二阶段总结：提取记忆节点...")
                events_list = [e if isinstance(e, dict) else e.model_dump() for e in summary.events]
                nodes_result = await self.extract_nodes_from_events(events_list, date_str, existing_nodes_context=existing_nodes_context)
                if nodes_result:
                    summary.nodes, summary.deleted_nodes = nodes_result

            return summary

        except Exception as e:
            logger.error(f"两阶段总结过程中发生错误: {e}", exc_info=True)
            return summary # 返回部分结果
