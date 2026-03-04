# AstrBot本地回忆插件[APLR]
# AstrBot Local Reminiscence Plugin

轻量级本地记忆插件，使用本地Embedding模型、本地数据库存储和回忆聊天记录。无需额外API密钥，零Embedding成本，节约Token，完全保护隐私。
使用Cron Job自动记录对话，通过深度学习语义搜索帮助AI自动回忆起相关经历。

A lightweight local memory plugin for AstrBot that uses local embedding models and local database storage to save and recall chat history. No API keys required, zero embedding costs, token-saving, and complete privacy protection. Automatically records conversations using Cron jobs, and helps AI automatically recall relevant experiences through deep learning semantic search.

(另：如果你想把这个插件改成适配其他平台的插件，应该只要改main.py就好了)



## ✨ 功能特性 / Features

- **本地化与隐私安全 / Local & Private**
  使用本地 Embedding 模型和本地 SQLite 数据库，无需额外 API，零成本且保护隐私。
  Uses local embedding models and SQLite database. No external APIs required, zero cost, and privacy-focused.

- **每日总结 / Daily Summarization**
  自动或手动将聊天记录总结为结构化的事件和每日感悟，并向量化存储事件。
  Automatically or manually summarizes chat history into structured events and daily reflections, and stores events as vectors.

- **记忆节点提取 / Memory Node Extraction**
  识别并存储重要的实体（人物、地点、概念），包含描述和类型。
  Identifies and stores important entities (people, places, concepts) with descriptions and types.

- **新对话自动历史注入 / Auto-History Injection**
  检测到开启新对话时自动注入三天内重要事件和七天内大致总结，保持长期的对话一致性。
  Automatically injects important events from the last 3 days and general summaries from the last 7 days when a new conversation is detected, maintaining long-term consistency.

- **聊天时自动回忆 / Auto-Recall During Chat**
  聊天时自动把记忆节点中存在的概念和过往的相关记忆注入系统 Prompt。
  Automatically injects existing concepts from memory nodes and relevant past memories into the System Prompt during chat.
---

## 🛠️ 使用指南 / How to use

使用本插件必须执行前三个步骤！
To use this plugin, you must perform the first three steps!


1.  **下载安装 / Installation**
    - 从Astrbot插件市场搜索下载此插件
    - 注意第一次下载（加载）本插件时需要有办法（魔法）链接至HuggingFace以便下载模型！
    - 下载时会自动安装依赖，可能需要等一会

2.  **配置插件 / Configuration**:
    - 到配置界面把core地址改成自己Astrbot的core目录
    - 设置目标用户ID用于识别对话，`机器人ID:FriendMessage:会话ID`，相关对话会用于每日总结。注：目前建议使用私聊（FriendMessage）或平台对话，因为作者的AI还没群聊过，当前代码用于群聊时，AI容易分不清群友谁是谁！
    - 设置用户名称，在AI的记忆中你（设置的目标用户）使用的名字
    - 其他配置不太重要，有需要就改

3.  **设置每日总结 / Daily Summary**
    - 定时任务方式：比如你每天通常在晚上11点50关机，就告诉你的AI“设一个定时任务，每天晚上11点45使用工具daily_summary_tool，日期参数写当天日期”（具体表述随意）
    - 手动触发方式：每天晚上关机前在AI聊天框输入 /daily_summary_command YYYY-MM-DD (此处日期是今日日期)
    - 上面两种方式二选一

4.  [可选] **系统prompt额外提醒 / Add in system_prompt**
    - 如果你的AI平时傻傻的想不起来调用工具，可以考虑在人设文本里加一句“你可以使用recall_memory_tool工具回忆与输入文本相关的记忆，可以使用recall_node_tool工具回忆特定概念”
    - 如果你的AI很聪明，或者不需要它经常调用工具就不用啦，本插件本来也有自动回忆功能的！

---

## ⌨️ 指令 / Commands

- `/daily_summary_command [YYYY-MM-DD]`
  手动触发指定日期的总结（默认为今天）。
  Manually trigger a summary for a specific date (defaults to today).

- `/recall_memory_command [text] [count]`
  根据输入文本手动搜索相关记忆。
  Manually search for relevant memories based on input text.

- `/recall_node_command [name]`
  搜索特定的记忆节点（人物、地点或概念）。
  Search for a specific memory node (person, place, or concept).

- `/update_nodes_command [YYYY-MM-DD]`
  从指定日期的已有事件中提取记忆节点，无需重新总结。本指令通常不需要使用。
  Extract memory nodes from existing events of a specific date without re-summarizing.

- `/vectorize_events [YYYY-MM-DD]`
  将指定日期的事件向量化并存入向量数据库。本指令通常不需要使用。
  Vectorize events of a specific date and store them in the vector database.

---

## 🔬 技术亮点 / Technical Highlights

- **Embedding 模型后台常驻 / Background Embedding Loading**
  模型在后台持续加载，调用时无需等待，极速响应（AstrBot 整体运行内存占用约 1.5G）。
  The model stays loaded in the background for instant response without waiting (AstrBot overall memory usage is ~1.5G).

- **多维权重排序 / Multi-dimensional Ranking**
  回忆检索不仅看语义相关度，还综合考虑了事件的重要性、情感强度以及随时间衰减的权重。动态计算搜索结果中的关键词重要性，优先考虑稀有词汇。
  Recall retrieval considers semantic relevance, event importance, emotional intensity, and time decay weights. Dynamically calculates keyword importance within the search results to prioritize rare terms.

- **AI 主动回忆工具 / Active Recall Tools**
  提供 `recall_node_tool` 和 `recall_memory_tool`，使 AI 能够根据对话需要主动进行背景查询和往事联想。
  Provides `recall_node_tool` and `recall_memory_tool`, enabling the AI to actively perform background queries and associations based on conversation needs.

---
## 流程图

![APLR流程图](./mermaid-diagram-.png)

## 📄 To Do List

- 目前每日总结仅适用于私聊，回忆功能倒是哪里都能用，将在下版本把每日总结改成适用群聊
- 节点关联
