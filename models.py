from pydantic import BaseModel, Field
from typing import List

class Event(BaseModel):
    event_id: str = Field(description="唯一ID，格式 evt_YYYYMMDD_序号")
    narrative: str = Field(description="事件完整叙述，完整主谓宾")
    emotion: str = Field(description="{ai_name}的情绪反应")
    importance: int = Field(ge=1, le=10, description="重要性 1-10，1是不重要，10是非常重要；需要注意每个个体面对同一件事情的时候所认为的重要程度都是不一样的，按照自己的真实想法填写即可")
    emotional_intensity: int = Field(ge=1, le=10, description="情绪强度 1-10，1是很平静，10是情绪非常强烈；需要注意每个个体面对同一件事情的时候情绪反应都是不一样的，按照自己的真实感受填写即可")
    tags: List[str] = Field(description="标签数组。请优先从以下核心分类中选择：[生活, 情感, 成长, 琐事, 技术, 灵感, 健康, 友谊]。如有需要，可在此基础上自由扩展更具体的子标签。")

class MemoryNode(BaseModel):
    name: str = Field(description="节点名称（实体或概念，如：王小美、火锅、考研）")
    type: str = Field(description="节点类型（如：人物、食物、活动、情感、地点、技术）")
    description: str = Field(description="对该节点的定义或最新状态描述（如：olozhika的好友，性格豪爽，最近在备考）")

class DailySummary(BaseModel):
    date: str = Field(description="日期 YYYY-MM-DD")
    events: List[Event] = Field(description="事件列表")
    nodes: List[MemoryNode] = Field(description="从今日对话中提取或更新的记忆节点（实体/概念）")
    daily_reflection: str = Field(description="{ai_name}的自由总结/心得")
