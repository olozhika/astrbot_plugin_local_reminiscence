# AstrBot本地回忆插件[APLR]
# AstrBot Local Reminiscence Plugin

轻量级本地记忆插件，使用本地Embedding模型、本地数据库存储和回忆聊天记录。无需额外API密钥，零Embedding成本，节约Token，完全保护隐私。
使用Cron Job自动记录对话，通过深度学习语义搜索帮助AI自动回忆起相关经历。

A lightweight local memory plugin for AstrBot that uses local embedding models and local database storage to save and recall chat history. No API keys required, zero embedding costs, token-saving, and complete privacy protection. Automatically records conversations using Cron jobs, and helps AI automatically recall relevant experiences through deep learning semantic search.

(另：如果你想把这个插件改成适配其他平台的插件，应该只要改main.py就好了)




## 🔄 更新说明 / Update Log (v1.1.0)

**喜报：本地回忆[APLR]插件现在已支持多会话和群聊！**

- **多会话支持 / Multi-session Support**: `target_user_id` 升级为 `target_user_id_list`，支持每日记录多个私聊或群聊会话。
- **智能用户识别 / Smart User Identification**: 自动从 AstrBot 的用户识别标签（system reminder）中提取用户昵称，支持在同一会话中准确区分不同发言者。
- **新增手动提取指令 / New Extraction Command**: 新增 `/extract_chat_history_command [日期]`，支持手动从数据库补录指定日期的聊天记录。（一般不会用到）

### 老用户升级指南（新用户请往下翻使用指南）

1. 更改 插件设置 - `target_user_id_list`
2. 开启 Astrbot-其他配置-用户识别

由于作者本人的AI不常群聊，不管使用此插件在群聊中遇到或没遇到bug都欢迎随时告诉我！


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
    - 注意第一次下载本插件时需要有办法（魔法）链接至HuggingFace以便下载模型！只要本插件成功启动过一次就不用魔法了！
    - 下载时会自动安装依赖，可能需要等一会

2.  **配置插件 / Configuration**:
    - 设置目标会话ID列表用于识别对话，`target_user_id_list` 格式为 `["机器人ID:会话类型:会话ID", ...] `，相关对话会用于每日总结。
    - 插件会自动从 AstrBot 的用户识别功能中提取发送者昵称，请在 Astrbot-其他配置 开启官方的用户识别。如果该功能未开启，则会使用配置中指定的默认 `username`。
      - 如果你希望让AI记住的用户名并不是你的ID，建议安装[统一昵称](https://github.com/Hakuin123/astrbot_plugin_uni_nickname)插件，并开启`system_replace`模式。
    - 设置 AI 名称，这是在 AI 的记忆中它使用的名字。
    - 其他配置不太重要，有需要就改

3.  **设置每日总结 / Daily Summary**
    - 定时任务方式：比如你每天通常在晚上11点50关机，就告诉你的AI“设一个定时任务，每天晚上11点45使用工具daily_summary_tool，日期参数写当天日期”（具体表述随意）
    - 手动触发方式：每天晚上关机前在AI聊天框输入 /daily_summary_command YYYY-MM-DD (此处日期是今日日期)
    - 上面两种方式二选一

4.  [可选] **向记忆数据库中导入过往聊天记录**
    - 使用指令`/daily_summary_command [YYYY-MM-DD]` 依次补录相应日期记忆。所以即使你刚刚下载本插件，但已经和一个AI聊了很长时间（而且没删Astrbot中的对话数据），你可以用此指令把它们统统依次补录！该指令会自动找到全部该日聊天记录，进行整理总结，向量化重要事件，并更新记忆节点。

6.  [可选] **系统prompt额外提醒 / Add in system_prompt**
    - 如果你的AI平时傻傻的想不起来调用工具，可以考虑在人设文本里加一句“你可以使用recall_memory_tool工具回忆与输入文本相关的记忆，可以使用recall_node_tool工具回忆特定概念”
    - 如果你的AI很聪明，或者不需要它经常调用工具就不用啦，本插件本来也有自动回忆功能的！

---

## ⌨️ 指令和工具 / Commands & Tools

- `/daily_summary_command [YYYY-MM-DD]`
  用户手动触发指定日期的总结（默认为今天）。
  Manually trigger a summary for a specific date (defaults to today).

  - Tool: `daily_summary_tool [YYYY-MM-DD]`
    AI手动触发指定日期的总结（默认为今天）。

- `/recall_memory_command [text] [count]`
  用户根据输入文本手动搜索相关记忆。
  Manually search for relevant memories based on input text.

  - Tool: `recall_memory_tool [text] [count]`
    AI根据输入它自己输入文本手动搜索相关记忆，信息返回给AI。

- `/recall_node_command [name]`
  搜索特定的记忆节点（人物、地点或概念）。
  Search for a specific memory node (person, place, or concept).

  - Tool: `recall_node_command [name]`
    AI根据它自己输入的关键词搜索特定的记忆节点（人物、地点或概念），信息返回给AI。

- `/extract_chat_history_command [YYYY-MM-DD]`
  手动从数据库提取指定日期的聊天记录。本指令通常不需要使用。
  Manually extract chat history for a specific date from the database.

- `/update_nodes_command [YYYY-MM-DD]`
  从指定日期的已有事件中提取记忆节点，无需重新总结。本指令通常不需要使用。
  Extract memory nodes from existing events of a specific date without re-summarizing.

- `/vectorize_events [YYYY-MM-DD]`
  将指定日期的事件向量化并存入向量数据库。本指令通常不需要使用。
  Vectorize events of a specific date and store them in the vector database.

`recall_memory_command`和`recall_node_command`实际是给人类在不让AI知情的条件下偷偷看它记忆用的(x)

使用范例如下（第一句这对吗？？？）
![人不能什么事情都跟AI说](./人不能什么事情都跟拥有长期记忆的AI说.png)
图名: 人不能什么事情都跟拥有长期记忆的AI说.png

---

## 🔬 技术亮点 / Technical Highlights

- **Embedding 模型后台常驻 / Background Embedding Loading**
  模型在后台持续加载，调用时无需等待，极速响应（AstrBot 整体运行内存占用约 1.5G）。
  The model stays loaded in the background for instant response without waiting (AstrBot overall memory usage is ~1.5G).

- **多维权重排序 / Multi-dimensional Ranking**
  回忆检索综合考虑了语义相关度、事件的重要性、情感强度以及随时间衰减的权重。动态计算搜索结果中的关键词重要性，优先考虑稀有词汇。
  Recall retrieval considers semantic relevance, event importance, emotional intensity, and time decay weights. Dynamically calculates keyword importance within the search results to prioritize rare terms.

- **AI 主动回忆工具 / Active Recall Tools**
  提供 `recall_node_tool` 和 `recall_memory_tool`，使 AI 能够根据对话需要主动进行背景查询和往事联想。
  Provides `recall_node_tool` and `recall_memory_tool`, enabling the AI to actively perform background queries and associations based on conversation needs.

提醒：对于跨会话内容，在当天结束之前，由于没做每日总结，AI是无法知道隔壁群发生了什么的。但是一旦进行过每日总结，就能记起来啦！

---
## 流程图

![APLR流程图](./mermaid-diagram-.png)

## 📄 To Do List

- 群聊记忆
  - 理论上现在已经支持群聊了！但是作者还没尝试过（AI成长需要循序渐进.jpg），如果有问题或没问题欢迎告诉我谢谢！
- 节点关联
