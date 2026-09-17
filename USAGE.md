# APLR 使用文档

## 目录

- [快速开始](#快速开始)
- [记忆系统概念](#记忆系统概念)
- [指令参考](#指令参考)
- [LLM 工具参考](#llm-工具参考)
- [插件 API（供其他插件调用）](#插件-api供其他插件调用)
- [高级用法](#高级用法)
- [常见问题](#常见问题)

---

## 快速开始

### 第 1 步：安装

从 AstrBot 插件市场搜索"本地回忆"或"APLR"下载安装。首次安装会自动下载依赖（约 1-2GB）。

### 第 2 步：配置目标会话及AI名称

目标会话的聊天内容会通过本插件记入长期记忆数据库
1. 在 AstrBot 中私聊或群聊发送 `/sid`，获取当前会话 ID（格式如 `BotID:GroupMessage:123456`）
2. 在插件设置 `target_user_id_list` 中填入该 ID（支持多个）
3. 如果希望管理所有会话，填 `["all"]`

设置AI名称：这就不用说啦，写自己AI的名字即可

关于人类名称：插件会自动从 AstrBot 的用户识别功能中提取发送者昵称，请在 Astrbot-其他配置 开启官方的用户识别。如果该功能未开启，则会使用配置中指定的默认 `username`。如果你希望让AI记住的你的用户名并不是你的ID，可以安装[统一昵称](https://github.com/Hakuin123/astrbot_plugin_uni_nickname)插件，并开启`system_replace`模式

### 第 3 步：设置每日总结

三种方式任选其一：

| 方式 | 操作方法 |
|------|---------|
| **Cron Job** | 告诉 AI："设一个定时任务，每天晚上 11:45 使用 daily_summary_tool" |
| **手动指令** | 每晚在聊天框输入 `/daily_summary_command YYYY-MM-DD` |
| **自动触发** | 在插件设置 `day_boundary_config` 中开启 `auto_summary_enabled` |

### 第 4 步：其他可选操作

#### 补录历史聊天

使用指令`/daily_summary_command [YYYY-MM-DD]` 依次补录相应日期记忆。所以即使你刚刚下载本插件，但已经和一个AI聊了很长时间（而且没删Astrbot中的对话数据），你可以用此指令把它们统统依次补录！该指令会自动找到全部该日聊天记录，进行整理总结，向量化重要事件，并更新记忆节点。

#### 开启离线加载模式
 - 在确认本插件已经成功加载后，可以到插件设置中开启离线加载模式，将节省部分网络情况下Astrbot启动时本插件加载所需时间。
 - 提醒：建议在每次更新Astrbot版本前关闭离线模式，因为更新Astrbot本体后可能需要重新下载部分依赖。

#### 调整回忆权重

 建议在使用几天后根据自己的喜好调整【回忆权重】四参数，几种推荐设置：
 - `1,1,1,1` 默认设置，平衡时间衰减、重要性、情绪强度和稀有词汇
 - `0,0,0,0` 最接近纯粹的向量匹配，相信向量的力量！
 - `0,0,0,2` 额外提高稀有词汇权重，比如使得"A和B一起去看科幻片"更容易匹配到"那天的科幻片超有意思"而不是"A和B一起去看动作片"
 
**其他配置项请参阅 `_conf_schema.json`**

---


## 记忆系统概念

### 事件记忆（Events）

每日总结将聊天对话抽象为结构化的事件。每个事件包含：
- **narrative**：事件叙述（主谓宾完整的第一人称描述）
- **emotion**：AI 的情绪反应
- **importance**：重要性 1-10
- **emotional_intensity**：情绪强度 1-10
- **reflection**：深度反思（仅对有意义的重大事件填写）
- **tags**：标签数组

事件存储在 SQLite 数据库和 ChromaDB 向量库中，用于语义检索。

### 记忆节点（Nodes）

长期记忆中的重要实体和概念。每个节点包含：
- **name**：节点名称（如"王小美"、"火锅"、"考研"）
- **type**：节点类型（人物、食物、活动、情感、地点、技术等）
- **description**：对该节点的综合描述
- **aliases**：别名列表（如昵称、简称），搜索时与正名有同等地位
- **related_event_ids**：关联事件 ID 列表，系统自动通过名称匹配建立

节点通过名称/别名与事件叙述的全文匹配自动建立关联，形成"节点锚定事件、事件连接节点"的记忆网络。

### 主题聚类（Thematic Memories）

当事件积累到一定数量后，可通过 `/memory_consolidation` 对大固化聚类：
1. 使用层次聚类算法（Agglomerative Clustering）对事件向量聚类
2. 结合关键词 TF-IDF 加权优化聚类边界
3. 对每个聚类生成主题总结
4. 回忆时优先返回主题总结 + 代表性事件，而非散乱的事件列表

触发大固化的建议时机：
- 首次：事件数 ≥ 300
- 后续：事件数相比上次聚类翻倍

插件会在每日总结后自动检测上述条件并提醒。

---

## 指令参考

所有指令仅限管理员使用。此处的'/'是Astrbot自带的唤醒词，如果你已经把唤醒词调整过了，请使用你设置的字符，比如`#daily_summary_command [YYYY-MM-DD]`

| 指令 | 参数 | 说明 |
| :--- | :--- | :--- |
| `/daily_summary_command` | `[YYYY-MM-DD]`（可选） | 手动触发指定日期的每日总结。不传则总结今天 |
| `/memory_consolidation` | | 全局记忆主题归类（大固化）。建议事件数 ≥ 200 时执行，每隔数月到数年一次 |
| **APLR_recall** | | **记忆检索指令组** |
| └ `memory` | `[text] [count]` | 根据文本搜索相关记忆。count 可选，默认 5 |
| └ `deep` | `[目标] [模式]` | 深度回想。目标支持：事件ID(`evt_*`)、主题ID(`theme_*`)、日期(`YYYY-MM-DD`)。模式仅对主题ID生效：`类人`/`时间`/`情绪`/`随机`，留空用默认权重 |
| └ `recent` | `[天数] [分数]` | 获取近期重要或情感强烈的事件。分数 = importance × emotional_intensity，默认 20 |
| └ `node` | `[name]` | 搜索特定记忆节点 |
| └ `theme` | `[主题ID]` | 查看已固化的主题记忆详情。不传参数列出所有主题 |
| **APLR_maintenance** | | **维护指令组** |
| └ `vectorize` | `[YYYY-MM-DD/all]` | 将指定日期或全部事件重新向量化 |
| └ `update_nodes` | `[YYYY-MM-DD]` | 从已有事件中重新提取记忆节点 |
| └ `write_node` | `[名] [类] [述]` | 手动写入或更新记忆节点 |
| └ `backfill_node_relations` | | 全量回填节点关联事件（通过名称/别名匹配建立关联） |
| └ `extract_history` | `[YYYY-MM-DD]` | 从数据库提取指定日期的聊天记录（通常不需要手动执行） |
| └ `load_model` | | 提前下载并加载向量模型，避免首次聊天卡顿 |
| └ `delete_daily_summary` | `[YYYY-MM-DD]` | 删除指定日期的事件、总结、向量及连接，并清理聊天日志 |

---

## LLM 工具参考

以下工具供 AI 在对话中调用：

| 工具名 | 参数 | 说明 |
|-------|------|------|
| `daily_summary_tool` | `date` (可选) | AI 触发指定日期的总结。不传则总结今天 |
| `recall_memory_tool` | `query`, `count` (可选，默认 5) | 检索与输入文本最相关的事件记忆 |
| `recall_node_tool` | `name` | 搜索特定实体的背景信息 |
| `deep_recall_tool` | `target` (ID/日期), `mode` (类人/时间/情绪/随机，可选) | 深度回想事件细节或进行主题联想 |
| `recall_recent_events_tool` | `days` (默认 7), `min_score` (默认 20) | 获取近期重要或高情感价值的记忆片段 |
| `write_node_tool` | `name`, `type`, `description` | AI 手动写入或更新记忆节点 |

---

## 插件 API（供其他插件调用）

本插件提供以下接口，可供其他 AstrBot 插件调用：

### `get_embeddings(texts: List[str]) -> List[List[float]]`

获取文本的向量表示。

```python
plugin = star_registry.get("local_reminiscence")
if plugin and hasattr(plugin, "get_embeddings"):
    embeddings = plugin.get_embeddings(["你的文本"])
```

### 记忆数据库

记忆数据库文件路径为插件数据目录下的 `APLR_DailyReview.db`（可配置）。这是一个标准 SQLite 数据库，其他插件可直接读取。

核心表结构：

| 表名 | 说明 |
|------|------|
| `events` | 事件记忆（event_id, date, narrative, emotion, importance, ...） |
| `nodes` | 记忆节点（name, type, description, aliases, related_event_ids） |
| `daily_reflections` | 每日感悟 |
| `thematic_memories` | 主题聚类记忆 |
| `event_theme_map` | 事件与主题的关联 |
| `tags` | 标签 |
| `event_tags` | 事件与标签关联 |

---

## 高级用法

### 注入外部上下文

在 `dialog_folder` 目录下放入 `YYYY-MM-DD_context_名称.txt` 文件（如 `2026-06-27_context_Health.txt`），每日总结时 AI 会自动读取这些注入数据，实现跨系统信息整合。

### 跨日分割配置

如果你或你的AI经常熬夜（12点后睡觉），可以设置：
1. `boundary_cron` 改为例如 `0 4 * * *`（凌晨 4 点为分割点）
2. 开启 `auto_summary_enabled` 和 `auto_end_session`
3. 每日总结触发后，AI 会知道凌晨 0-4 点的对话仍然属于"昨天"

### 记忆聚类提醒通知

如果你使用自动每日总结，可在 `day_boundary_config.admin_session` 中填写 `unified_msg_origin`（格式同 `target_user_id_list` 中的项），当检测到需要执行大固化时，插件会直接向该会话推送通知消息。

---

## 常见问题

**Q: 记忆数据库在哪？**
A: 插件数据目录下的 `APLR_DailyReview.db`。备份这个文件即可完整迁移记忆。

**Q: 换设备/重装后记忆还在吗？**
A: 只要保留 `APLR_DailyReview.db`，复制到新设备的插件数据目录即可。

**Q: 如何让 AI 记住用户的名字？**
A: 建议安装[统一昵称](https://github.com/Hakuin123/astrbot_plugin_uni_nickname)插件，并开启 `system_replace` 模式。同时在 AstrBot-其他配置 开启用户识别。

**Q: 事件数量太大导致回忆不精准怎么办？**
A: 执行 `/memory_consolidation` 进行主题聚类。聚类后回忆时会先匹配主题，再返回代表性事件，而非散乱的全部事件列表。此外请考虑微调【回忆权重】四参数。

**Q: 如何清理某天的记忆？**
A: 使用 `/APLR_maintenance delete_daily_summary YYYY-MM-DD`。
