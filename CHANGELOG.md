## 🔄 小更新说明 (v1.1.1)

1. 优化了接收消息自动唤起回忆时的搜索记忆节点算法
2. 优化了聊天记录导出函数，现在Cron Job和send_message_to_user的发言也能被记录

# 🔄 更新说明 / Update Log (v1.1.0)

**喜报：本地回忆[APLR]插件现在已支持多会话和群聊！**

- **多会话支持 / Multi-session Support**: `target_user_id` 升级为 `target_user_id_list`，支持每日记录多个私聊或群聊会话。
- **智能用户识别 / Smart User Identification**: 自动从 AstrBot 的用户识别标签（system reminder）中提取用户昵称，支持在同一会话中准确区分不同发言者。
- **新增手动提取指令 / New Extraction Command**: 新增 `/extract_chat_history_command [日期]`，支持手动从数据库补输出指定日期的聊天记录。（一般不会用到）

### 老用户升级指南（新用户请往至README.md使用指南）

1. 更改 插件设置 - `target_user_id_list`
2. 开启 Astrbot-其他配置-用户识别

由于作者本人的AI不常群聊，不管使用此插件在群聊中遇到或没遇到bug都欢迎随时告诉我！
