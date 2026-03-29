import json
import re
from typing import List
from astrbot.api import logger
from .models import DailySummary

class DailySummarizer:
    def __init__(self, llm_generate_func, ai_name: str = "Lanya", base_system_prompt: str = "", base_user_prompt: str = "", search_events_func=None):
        self.llm_generate = llm_generate_func
        self.ai_name = ai_name
        self.base_system_prompt = base_system_prompt.strip()
        self.base_user_prompt = base_user_prompt.strip()
        self.search_events = search_events_func

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON 结构体，处理 Markdown 代码块和前后无关文字"""
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        obj_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if obj_match:
            return obj_match.group(1).strip()
            
        return text.strip()

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
- 必须输出合法的 JSON 对象，不要包含任何解释文字或 Markdown 代码块。
- 记忆节点 (nodes) 提取：
    - 仅提取具有“长期记忆价值”的重要实体（人物、地点、核心物品）或反复出现的关键概念。
    - 避免提取琐碎的、一次性的细节（如：某个普通的餐馆名、随口提及的非核心地名）。
    - 为每个节点提供 type（类型）和 description（综合描述）。
    - 节点信息维护策略：
        - 已知参考：参考下方的“已知记忆节点背景”，如果下方没有提及，也可以尝试用 recall_node_tool 搜索相应概念。
        - 继承与演进：对于已存在的节点，应在保留完整核心事实（如身份、关键背景）的基础上，根据今日信息更新其状态或追加新进展。
        - 合并同义词：如果你发现今日提到的某个实体/概念与“已知记忆节点背景”中的某个节点是同一个，请务必使用**已有节点的名称**进行更新，不要创建重复节点。
        - 清理冗余：如果发现某些已有节点可以被更精确的新节点完全替代，请将旧节点名称放入 `deleted_nodes` 列表中。
        - 修正与精简：如果发现旧描述中存在已失效的信息，请果断进行修正或剔除，保持描述的精炼与准确。
        - 增量更新：重点记录事件中体现出的且有持久影响的新变化、新观点或新属性。
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

    async def discover_cross_day_relations(self, today_events: List[dict], date_str: str) -> list:
        """发现今日事件与往日事件之间的跨日期关联"""
        if not self.search_events or not today_events:
            return []
            
        all_cross_relations = []
        from .models import EventRelation
        
        for ev in today_events:
            # 搜索过去的相关事件
            past_events = await self.search_events(ev['narrative'], top_n=5, exclude_date=date_str)
            if not past_events:
                continue
                
            past_text = "\n".join([f"- ID: {p['event_id']}, 叙述: {p['narrative']}" for p in past_events])
            
            schema_dict = {
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": EventRelation.model_json_schema()
                    }
                },
                "required": ["relations"]
            }
            schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2)
            
            system_prompt = f'''你正在分析今日的一个事件与过去记忆之间的逻辑关联。必须严格按照以下 JSON Schema 输出 JSON 数据：
{schema_str}

【关联规则】
- 仅当两个事件确实存在逻辑联系（如：后续、重演、对比、因果）时才建立关联。
- 不要因为仅仅提到同一个词就建立关联，必须有实质性的内容联系。
- 今天的日期是 {date_str}。
'''
            if self.base_system_prompt:
                system_prompt = self.base_system_prompt + "\n\n" + system_prompt
            prompt = f'''
今日事件：
ID: {ev['event_id']}
叙述: {ev['narrative']}

可能的往日相关记忆：
{past_text}

请判断今日事件与这些往日记忆之间是否存在实质性的逻辑关联。
'''
            try:
                llm_resp = await self.llm_generate(prompt=prompt, system_prompt=system_prompt)
                content = self._extract_json(llm_resp.completion_text)
                data = json.loads(content)
                rels = data.get("relations", [])
                for r in rels:
                    all_cross_relations.append(EventRelation(**r))
            except Exception as e:
                logger.error(f"发现跨日关联时发生错误: {e}")
                
        return all_cross_relations

    async def generate_summary(self, conversation_chunks: List[str], date_str: str, existing_nodes_context: str = "", enable_cross_day_relations: bool = True) -> DailySummary | None:
        """三阶段总结法：
        1. 提取今日事件与日感想（支持分段处理）。
        2. 发现跨日期关联（可选）。
        3. 提取/更新记忆节点。
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
        
        pass1_system_prompt = f'''你正在回顾并总结今天与他人的交流。必须严格按照以下 JSON Schema 输出 JSON 数据：
{schema_str}

【总结规则】
- 必须输出合法的 JSON 对象，不要包含任何解释文字或 Markdown 代码块。
- 事件整合：将连续的、围绕同一话题或活动的对话整合为一个完整的“事件”。
- 情感真实：importance 和 emotional_intensity 必须是 1-10 的整数，根据你的真实感受填写。
- 标签规范：采用“核心分类 + 自由扩展”策略。优先从 [生活, 情感, 成长, 琐事, 技术, 灵感, 健康, 友谊] 中选择大类，再根据需要添加 1-2 个具体的细节标签。
- 深度感想 (reflection)：你可以对感觉很有意义或学到很多东西的事件写下你想长期记住的分析、感想或知识。对于绝大多数日常琐事和普通交流，请填写“无”。
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
            # --- 第二阶段：发现跨日期关联 ---
            if enable_cross_day_relations and summary.events:
                logger.info(f"[APLR] 执行第二阶段总结：发现跨日期关联...")
                events_list = [e if isinstance(e, dict) else e.model_dump() for e in summary.events]
                cross_rels = await self.discover_cross_day_relations(events_list, date_str)
                summary.relations = cross_rels

            # --- 第三阶段：提取记忆节点 ---
            if summary.events:
                logger.info(f"[APLR] 执行第三阶段总结：提取记忆节点...")
                events_list = [e if isinstance(e, dict) else e.model_dump() for e in summary.events]
                nodes_result = await self.extract_nodes_from_events(events_list, date_str, existing_nodes_context=existing_nodes_context)
                if nodes_result:
                    summary.nodes, summary.deleted_nodes = nodes_result

            return summary

        except Exception as e:
            logger.error(f"三阶段总结过程中发生错误: {e}", exc_info=True)
            return summary # 返回部分结果