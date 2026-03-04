import json
import logging
import re
from typing import List
from .models import DailySummary

logger = logging.getLogger(__name__)

class DailySummarizer:
    def __init__(self, llm_generate_func, base_system_prompt: str = "", base_user_prompt: str = ""):
        self.llm_generate = llm_generate_func
        self.base_system_prompt = base_system_prompt.strip()
        self.base_user_prompt = base_user_prompt.strip()

    async def extract_nodes_from_events(self, events: List[dict], date_str: str, existing_nodes_context: str = "") -> list | None:
        """从已有的事件叙述中提取记忆节点"""
        if not events:
            return []
            
        events_text = "\n---\n".join([f"事件ID: {ev['event_id']}\n叙述: {ev['narrative']}" for ev in events])
        
        from .models import MemoryNode
        schema = {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": MemoryNode.model_json_schema()
                }
            },
            "required": ["nodes"]
        }
        
        context_text = f"\n【已知记忆节点背景】\n{existing_nodes_context}\n" if existing_nodes_context else ""
        
        system_prompt = f"""你正在从一组你之前总结的事件中提取记忆节点。必须严格按照以下 JSON Schema 输出 JSON 数据：
{json.dumps(schema, ensure_ascii=False, indent=2)}

【提取规则】
- 必须输出合法的 JSON 对象，不要包含任何解释文字或 Markdown 代码块。
- 记忆节点 (nodes) 提取：
    - 仅提取具有“长期记忆价值”的重要实体（人物、地点、核心物品）或反复出现的关键概念。
    - 避免提取琐碎的、一次性的细节（如：某个普通的餐馆名、随口提及的非核心地名）。
    - 为每个节点提供 type（类型）和 description（综合描述）。
    - **节点信息维护策略**：
        - **继承与演进**：参考下方的“已知记忆节点背景”。对于已存在的节点，应在保留核心事实（如身份、关键背景）的基础上，根据今日信息更新其状态或追加新进展。
        - **修正与精简**：如果发现旧描述中存在已失效、不合时宜或过于琐碎的信息，请果断进行修正或剔除，保持描述的精炼与准确。
        - **增量更新**：重点记录今日对话中体现出的新变化、新观点或新属性。
    - 今天的日期是 {date_str}。
{context_text}
"""
        try:
            llm_resp = await self.llm_generate(
                prompt=f"请从以下事件中提取记忆节点：\n\n{events_text}",
                system_prompt=system_prompt,
            )

            content = llm_resp.completion_text.strip()
            content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^```\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content)
            
            data = json.loads(content)
            nodes_data = data.get("nodes", [])
            return [MemoryNode(**n) for n in nodes_data]

        except Exception as e:
            logger.error(f"提取节点时发生错误: {e}", exc_info=True)
            return None

    async def generate_summary(self, conversation_text: str, date_str: str, existing_nodes_context: str = "") -> DailySummary | None:
        schema = DailySummary.model_json_schema()
        context_text = f"\n【已知记忆节点背景】\n{existing_nodes_context}\n" if existing_nodes_context else ""
        
        core_system_prompt = f"""你正在回顾并总结今天与朋友的交流。必须严格按照以下 JSON Schema 输出 JSON 数据：
{json.dumps(schema, ensure_ascii=False, indent=2)}

【总结规则】
- 必须输出合法的 JSON 对象，不要包含任何解释文字或 Markdown 代码块。
- 事件整合：将连续的、围绕同一话题或活动的对话整合为一个完整的“事件”。
- 情感真实：importance 和 emotional_intensity 必须是 1-10 的整数，根据你的真实感受填写。
- 标签规范：采用“核心分类 + 自由扩展”策略。优先从 [生活, 情感, 成长, 琐事, 技术, 灵感, 健康, 友谊] 中选择大类，再根据需要添加 1-2 个具体的细节标签。
- 记忆节点 (nodes) 提取：
    - 仅提取具有“长期记忆价值”的重要实体（人物、地点、核心物品）或反复出现的关键概念。
    - 避免提取琐碎的、一次性的细节（如：某个普通的餐馆名、随口提及的非核心地名）。
    - 为每个节点提供 type（类型）和 description（综合描述）。
    - **节点信息维护策略**：
        - **继承与演进**：参考下方的“已知记忆节点背景”。对于已存在的节点，应在保留核心事实（如身份、关键背景）的基础上，根据今日信息更新其状态或追加新进展。
        - **修正与精简**：如果发现旧描述中存在已失效、不合时宜或过于琐碎的信息，请果断进行修正或剔除，保持描述的精炼与准确。
        - **增量更新**：重点记录今日对话中体现出的新变化、新观点或新属性。
- 格式规范：
    - event_id 格式：evt_{date_str.replace('-', '')}_序号（如 001, 002）。
    - 今天的日期是 {date_str}。
{context_text}
"""

        if self.base_system_prompt:
            system_prompt = self.base_system_prompt + "\n\n" + core_system_prompt
        else:
            system_prompt = core_system_prompt
        logger.info(f"[APLR] SYS_PROM: {system_prompt}")

        if self.base_user_prompt:
            user_prompt = self.base_user_prompt + "\n\n" + conversation_text
        else:
            user_prompt = conversation_text

        try:
            llm_resp = await self.llm_generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            content = llm_resp.completion_text.strip()
            content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^```\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content)
            content = content.strip()

            data = json.loads(content)
            summary = DailySummary(**data)

            if summary.date != date_str:
                logger.warning(f"返回的日期 {summary.date} 与期望的 {date_str} 不符，将强制覆盖")
                summary.date = date_str

            return summary

        except Exception as e:
            logger.error(f"生成总结时发生错误: {e}", exc_info=True)

        return None
