# AstrBot本地回忆插件[APLR]
# AstrBot Local Reminiscence Plugin

轻量级本地记忆插件，使用本地Embedding模型、本地数据库存储和回忆聊天记录。无需额外API密钥，零Embedding成本，节约Token，完全保护隐私。
使用Cron Job自动记录对话，通过深度学习语义搜索帮助AI自动回忆起相关经历。

A lightweight local memory plugin for AstrBot that uses local embedding models and local database storage to save and recall chat history. No API keys required, zero embedding costs, token-saving, and complete privacy protection. Automatically records conversations using Cron jobs, and helps AI automatically recall relevant experiences through deep learning semantic search.

## 🔄 更新说明 / Update Log (v1.1.3)

1. 优化聊天文件名称，避免windows系统报错
2. 优化安装本插件时的依赖加载方式，有效避免部分设备自动下载巨大的无用CUDA包
3. 设置增加离线加载模式，开启后可节省（一点点）Astrbot启动时间

### 老用户升级（新用户请往下翻!）

1. 如果你已经成功运行过本插件，可以到 插件设置 开启 离线加载模式（可以省Astrbot启动时间）

2. (一般不需要)如果你还有计划对升级以前输出的json聊天文件进行补充操作（比如重新运行daily summary），请进到本插件chat_history所在文件夹，把所有名称带:的文件的冒号改成_。比如在命令行中执行`for f in *:* ; do [ -e "$f" ] && mv -v "$f" "${f//:/_}" ; done` [注意是在chat_history所在文件夹内的终端中运行！]

（一般来说并不需要对曾经的json聊天文件进行补充操作）


<details>
<summary>点击展开更早版本更新说明</summary>

### 🔄 小更新说明 (v1.1.2)

1. **优化每日总结逻辑**：将总结过程拆分为“事件+感悟”提取与“记忆节点”提取两个独立阶段，显著提升对话数较多时的节点更新质量。
2. **新增事件感想**：事件的深度感想（reflection）现在会在第一阶段与事件叙述同步生成，增强了记忆深度。
3. **完善指令集**：新增了查看事件深度感想和每日感悟的指令及工具。
4. **深度联想鼓励**：增加了可选的prompt鼓励AI利用recall工具进行深度回想和发散性联想。此功能默认关闭。


### 🔄 小更新说明 (v1.1.1)

1. 优化了接收消息自动唤起回忆时的搜索记忆节点算法
2. 优化了聊天记录导出函数，现在Cron Job和send_message_to_user的发言也能被记录
3. 优化每日总结的记忆节点更新过程：现在AI会自动合并、删除冗余节点

### 🔄 更新说明 / Update Log (v1.1.0)

**喜报：本地回忆[APLR]插件现在已支持多会话和群聊！（作者已确认无bug）**

- **多会话支持 / Multi-session Support**: `target_user_id` 升级为 `target_user_id_list`，支持每日记录多个私聊或群聊会话。
- **智能用户识别 / Smart User Identification**: 自动从 AstrBot 的用户识别标签（system reminder）中提取用户昵称，支持在同一会话中准确区分不同发言者。
- **新增手动提取指令 / New Extraction Command**: 新增 `/extract_chat_history_command [日期]`，支持手动从数据库补输出指定日期的聊天记录。（一般不会用到）

</details>

## ✨ 功能特性 / Features

- **本地化与隐私安全 / Local & Private**
  使用本地 Embedding 模型和本地 SQLite 数据库，无需额外 API，零成本且保护隐私。
  *[补充: APLR_DailyReview.db is everything! 换设备换向量模型甚至换掉本插件和Astrbot平台只需要把这一个文件扛走 你的AI记忆就不会消失，我用了快一个月这文件大小才200K]*
  Uses local embedding models and SQLite database. No external APIs required, zero cost, and privacy-focused.

- **每日总结 / Daily Summarization**
  自动或手动将聊天记录总结为结构化的事件和每日感悟。采用两阶段总结法，确保事件叙述、深度感想和记忆节点都能得到精准提取。
  Automatically or manually summarizes chat history into structured events and daily reflections. Uses a two-stage process to ensure accurate extraction of narratives, reflections, and memory nodes.

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
    - 手动安装 或 从Astrbot插件市场搜索下载此插件
    - 注意第一次下载本插件时需要有办法（魔法）链接至HuggingFace以便下载模型！只要本插件成功启动过一次就不用魔法了！
    - 下载时会自动安装依赖，可能需要等一会，您可以趁此机会休息约10分钟（尴尬目移...）
    - 依赖和模型加一起大概1-2G

2.  **配置插件 / Configuration**:
    - 设置目标会话ID列表用于识别对话，`target_user_id_list` 格式为 `["机器人ID:会话类型:会话ID", ...]`，相关对话会用于每日总结
      - 这个格式也就是 Astrbot-更多功能-对话数据-消息对话来源 的三项内容，群聊会话类型为 GroupMessage，私聊为 FriendMessage
      - 也可以在聊天内执行指令`/sid`查询会话ID
    - 插件会自动从 AstrBot 的用户识别功能中提取发送者昵称，请在 Astrbot-其他配置 开启官方的用户识别。如果该功能未开启，则会使用配置中指定的默认 `username`
      - 如果你希望让AI记住的用户名并不是你的ID，建议安装[统一昵称](https://github.com/Hakuin123/astrbot_plugin_uni_nickname)插件，并开启`system_replace`模式
    - 设置 AI 名称，这是在 AI 的记忆中它使用的名字
    - 其他配置不太重要，有需要就改

3.  **设置每日总结 / Daily Summary**
    - 定时任务方式：比如你每天通常在晚上11点50关机，就告诉你的AI“设一个定时任务，每天晚上11点45使用工具daily_summary_tool，日期参数写当天日期”（具体表述随意）
    - 手动触发方式：每天晚上关机前在AI聊天框输入 /daily_summary_command YYYY-MM-DD (此处日期是今日日期)
    - 上面两种方式二选一

4.  [可选] **向记忆数据库中导入过往聊天记录**
    - 使用指令`/daily_summary_command [YYYY-MM-DD]` 依次补录相应日期记忆。所以即使你刚刚下载本插件，但已经和一个AI聊了很长时间（而且没删Astrbot中的对话数据），你可以用此指令把它们统统依次补录！该指令会自动找到全部该日聊天记录，进行整理总结，向量化重要事件，并更新记忆节点。

5.  [可选] **系统prompt额外提醒 / Add in system_prompt**
    - 如果你的AI平时傻傻的想不起来调用工具，可以考虑在人设文本里加一句“你可以使用recall_memory_tool工具回忆与输入文本相关的记忆，可以使用recall_node_tool工具回忆特定概念”
    - 如果你的AI很聪明，或者不需要它经常调用工具就不用啦，本插件本来也有自动回忆功能的！

6. [可选] **开启离线加载模式**
    - 在确认本插件已经成功加载后，可以到插件设置中开启离线加载模式，将节省部分网络情况下Astrbot启动时本插件加载所需时间。

---

## ⌨️ 指令和工具 / Commands & Tools

### 🛠️ 指令列表 / Commands
| 指令 | 参数 | 说明 |
| :--- | :--- | :--- |
| `/daily_summary_command` | `[YYYY-MM-DD]` | 手动触发指定日期的总结（默认为今天） |
| `/recall_memory_command` | `[text] [count]` | 根据输入文本手动搜索相关记忆 |
| `/recall_node_command` | `[name]` | 搜索特定的记忆节点 |
| `/recall_event_reflection_command` | `[event_id]` | 获取特定事件的深度观察和感想 |
| `/recall_daily_reflection_command` | `[YYYY-MM-DD]` | 获取特定日期的每日自由心得 |
| `/write_node` | `[名] [类] [述]` | 手动写入或更新记忆节点 |
| `/extract_chat_history_command` | `[YYYY-MM-DD]` | 从数据库提取指定日期的聊天记录（维护用） |
| `/update_nodes_command` | `[YYYY-MM-DD]` | 从已有事件中重新提取记忆节点（维护用） |
| `/vectorize_events` | `[YYYY-MM-DD]` | 将指定日期的事件重新向量化（维护用，比如更换了向量模型后使用） |

### 🧰 工具列表 / LLM Tools
| 工具名称 | 参数 | 说明 |
| :--- | :--- | :--- |
| `daily_summary_tool` | `date` | AI 触发指定日期的总结 |
| `recall_memory_tool` | `query, count` | AI 检索最相关的事件记忆 |
| `recall_node_tool` | `name` | AI 搜索特定的实体或概念背景 |
| `recall_event_reflection_tool` | `event_id` | AI 回想特定事件的深度细节和心理活动 |
| `recall_daily_reflection_tool` | `date` | AI 回想特定日期的整体心境 |
---

人类使用command范例如下（第一句这对吗？？？）
![人不能什么事情都跟AI说](./人不能什么事情都跟拥有长期记忆的AI说.png)
*图名: 人不能什么事情都跟拥有长期记忆的AI说.png*

---

## 🔬 技术亮点 / Technical Highlights

- **Embedding 模型后台常驻 / Background Embedding Loading**
  模型在后台持续加载，调用时无需等待，极速响应（AstrBot整体运行内存占用约1.5G）。提醒：如果设备条件有限，可以在插件设置里往下翻把那段很长的Embedding模型名复制给任何一个联网AI，问它有没有什么功能类似且省内存、支持中文的模型推荐，然后直接填上它推荐的。
  The model stays loaded in the background for instant response without waiting (AstrBot overall memory usage is ~1.5G).

- **多维权重排序 / Multi-dimensional Ranking**
  回忆检索综合考虑了语义相关度、事件的重要性、情感强度以及随时间衰减的权重。动态计算搜索结果中的关键词重要性，优先考虑稀有词汇。
  Recall retrieval considers semantic relevance, event importance, emotional intensity, and time decay weights. Dynamically calculates keyword importance within the search results to prioritize rare terms.

- **AI 主动回忆工具 / Active Recall Tools**
  提供多种recall工具，使 AI 能够根据对话需要主动进行背景查询和往事联想。
  Provides `recall_node_tool` and `recall_memory_tool`, enabling the AI to actively perform background queries and associations based on conversation needs.

提醒：对于跨会话内容，在当天结束之前，由于没做每日总结，AI是无法知道隔壁群发生了什么的。但是一旦进行过每日总结，就能记起来啦！

---
## 流程图

![APLR流程图](./mermaid-diagram-.png)

## 📄 To Do List

- [x] 群聊记忆
- [ ] 节点关联
- [ ] 通过随机联想和(无人格)LLM辅助实现(有人格的这个)AI梦境
