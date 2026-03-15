import json
import re
from typing import List
from astrbot.api import logger
from .models import DailySummary

class DailySummarizer:
    def __init__(self, llm_generate_func, ai_name: str = "Lanya", base_system_prompt: str = "", base_user_prompt: str = ""):
        self.llm_generate = llm_generate_func
        self.ai_name = ai_name
        self.base_system_prompt = base_system_prompt.strip()
        self.base_user_prompt = base_user_prompt.strip()

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
        
        system_prompt = f"""你正在从一组你之前总结的事件中提取记忆节点。必须严格按照以下 JSON Schema 输出 JSON 数据：
{schema_str}

【提取规则】
- 必须输出合法的 JSON 对象，不要包含任何解释文字或 Markdown 代码块。
- 记忆节点 (nodes) 提取：
    - 仅提取具有“长期记忆价值”的重要实体（人物、地点、核心物品）或反复出现的关键概念。
    - 避免提取琐碎的、一次性的细节（如：某个普通的餐馆名、随口提及的非核心地名）。
    - 为每个节点提供 type（类型）和 description（综合描述）。
    - **节点信息维护策略**：
        - **已知参考**：参考下方的“已知记忆节点背景”，如果下方没有提及，也可以尝试用 recall_node_tool 搜索相应概念。
        - **继承与演进**：对于已存在的节点，应在保留完整核心事实（如身份、关键背景）的基础上，根据今日信息更新其状态或追加新进展。
        - **合并同义词**：如果你发现今日提到的某个实体/概念与“已知记忆节点背景”中的某个节点是同一个，请务必使用**已有节点的名称**进行更新，不要创建重复节点。
        - **清理冗余**：如果发现某些已有节点可以被更精确的新节点完全替代，请将旧节点名称放入 `deleted_nodes` 列表中。
        - **修正与精简**：如果发现旧描述中存在已失效的信息，请果断进行修正或剔除，保持描述的精炼与准确。
        - **增量更新**：重点记录事件中体现出的且有持久影响的新变化、新观点或新属性。
    - 今天的日期是 {date_str}。
{context_text}
"""
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

    async def generate_summary(self, conversation_text: str, date_str: str, existing_nodes_context: str = "") -> DailySummary | None:
        """两阶段总结法：第一阶段提取事件/日感想（含事件感想），第二阶段提取记忆节点"""
        
        # --- 第一阶段：核心总结（事件、每日感悟） ---
        schema_dict = DailySummary.model_json_schema()
        
        # 移除 nodes 和 deleted_nodes，因为我们在第二阶段提取
        if "nodes" in schema_dict["properties"]:
            del schema_dict["properties"]["nodes"]
        if "deleted_nodes" in schema_dict["properties"]:
            del schema_dict["properties"]["deleted_nodes"]
        if "required" in schema_dict:
            schema_dict["required"] = [r for r in schema_dict["required"] if r not in ["nodes", "deleted_nodes"]]

        schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2).replace("{ai_name}", self.ai_name)
        
        pass1_system_prompt = f"""你正在回顾并总结今天与朋友的交流。必须严格按照以下 JSON Schema 输出 JSON 数据：
{schema_str}

【总结规则】
- 必须输出合法的 JSON 对象，不要包含任何解释文字或 Markdown 代码块。
- 事件整合：将连续的、围绕同一话题或活动的对话整合为一个完整的“事件”。
- 情感真实：importance 和 emotional_intensity 必须是 1-10 的整数，根据你的真实感受填写。
- 标签规范：采用“核心分类 + 自由扩展”策略。优先从 [生活, 情感, 成长, 琐事, 技术, 灵感, 健康, 友谊] 中选择大类，再根据需要添加 1-2 个具体的细节标签。
- 深度感想 (reflection)：针对每个事件写下你想长期记住的深入观察或个人感想。对于没有特别感想的日常琐事，reflection填“无”。
- 格式规范：
    - event_id 格式：evt_{date_str.replace('-', '')}_序号（如 001, 002）。
    - 今天的日期是 {date_str}。
"""
        
        full_system_prompt = (self.base_system_prompt + "\n\n" + pass1_system_prompt) if self.base_system_prompt else pass1_system_prompt
        user_prompt = (self.base_user_prompt + "\n\n" + conversation_text) if self.base_user_prompt else conversation_text

        try:
            logger.info("[APLR] 执行第一阶段总结：提取事件与日感想...")
            llm_resp1 = await self.llm_generate(prompt=user_prompt, system_prompt=full_system_prompt)
            content1 = self._extract_json(llm_resp1.completion_text)
            data1 = json.loads(content1)
            
            # 补全缺失字段以符合 DailySummary 模型
            data1["nodes"] = []
            if "deleted_nodes" not in data1:
                data1["deleted_nodes"] = []
                
            summary = DailySummary(**data1)

            if summary.date != date_str:
                logger.warning(f"返回的日期 {summary.date} 与期望的 {date_str} 不符，将强制覆盖")
                summary.date = date_str

            # --- 第二阶段：提取记忆节点 ---
            if summary.events:
                logger.info(f"[APLR] 执行第二阶段总结：从 {len(summary.events)} 个事件中提取记忆节点...")
                # 将 Event 对象转换为 dict 列表
                events_list = [e.model_dump() for e in summary.events]
                nodes_result = await self.extract_nodes_from_events(events_list, date_str, existing_nodes_context=existing_nodes_context)
                if nodes_result:
                    summary.nodes, summary.deleted_nodes = nodes_result

            return summary

        except Exception as e:
            logger.error(f"两阶段总结过程中发生错误: {e}", exc_info=True)
            return None
