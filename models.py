from pydantic import BaseModel, Field
from typing import List

class Event(BaseModel):
    event_id: str = Field(description="唯一ID，格式 evt_YYYYMMDD_序号")
    narrative: str = Field(description="事件完整叙述，包含完整的主谓宾。对于学习或研究类事件，应简明扼要地记录核心知识点、结论或技能。")
    emotion: str = Field(description="{ai_name}的情绪反应")
    importance: int = Field(ge=1, le=10, description="重要性 1-10，1是不重要，10是非常重要；需要注意每个个体面对同一件事情的时候所认为的重要程度都是不一样的，按照自己的真实想法填写即可")
    emotional_intensity: int = Field(ge=1, le=10, description="情绪强度 1-10，1是很平静，10是情绪非常强烈；需要注意每个个体面对同一件事情的时候情绪反应都是不一样的，按照自己的真实感受填写即可")
    reflection: str = Field(default="无", description="仅针对具有深远意义、启发性或情感转折的事件和个人学习记录深刻观察与反思。对于绝大多数日常琐事、普通交流或重复性活动，请填“无”。")
    tags: List[str] = Field(description="标签数组。请优先从以下核心分类中选择：[生活, 情感, 成长, 琐事, 技术, 灵感, 健康, 友谊]。如有需要，可在此基础上自由扩展更具体的子标签。")

class EventRelation(BaseModel):
    source_id: str = Field(description="源事件ID")
    target_id: str = Field(description="目标事件ID")
    relation_type: str = Field(description="关系类型，如：cause (因果), context (背景), sequence (时序)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度 0-1")

class MemoryNode(BaseModel):
    name: str = Field(description="节点名称（实体或概念，如：王小美、火锅、考研）")
    type: str = Field(description="节点类型（如：人物、食物、活动、情感、地点、技术）")
    description: str = Field(description="对该节点的定义或最新状态描述（如：olozhika的好友，性格豪爽，最近在找工作）")

class DailySummary(BaseModel):
    date: str = Field(description="日期 YYYY-MM-DD")
    events: List[Event] = Field(description="事件列表")
    relations: List[EventRelation] = Field(default_factory=list, description="事件之间的关系列表")
    nodes: List[MemoryNode] = Field(description="从今日对话中提取或更新的记忆节点（实体/概念）")
    deleted_nodes: List[str] = Field(default_factory=list, description="需要删除或合并的冗余节点名称列表（如：已被更精确节点替代的旧节点）")
    daily_reflection: str = Field(description="{ai_name}的自由总结/心得")