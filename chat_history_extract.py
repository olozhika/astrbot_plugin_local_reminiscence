import re
import os
import json
import sqlite3
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

def decode_json_unicode(s):
    try:
        return json.loads(s)
    except Exception:
        return s

def decode_unicode_escapes(s):
    """
    将字符串中的 \\uXXXX 转义序列转换为实际 Unicode 字符。
    例如: "\\u83b7" -> "获"
    """
    if not isinstance(s, str):
        return s
    # 使用正则匹配 \uXXXX 格式
    def replace_unicode(match):
        code = match.group(1)
        try:
            return chr(int(code, 16))
        except ValueError:
            return match.group(0)
    return re.sub(r'\\u([0-9a-fA-F]{4})', replace_unicode, s)

def get_date_key(ts):
    if ts is None:
        return "unknown_date"
    ts_norm = ts.replace(" ", "T")
    try:
        dt = datetime.fromisoformat(ts_norm)
        return dt.date().isoformat()
    except Exception:
        return ts.split(" ")[0].split("T")[0]

def clean_dialogue_with_different_limits(
        db_path: Path,
        output_dir: Path,
        username="olozhika",
        ai_name="Lanya",
        max_user_chars=1000, 
        max_assistant_chars=2000,
        platform="AstrBot",
        target_date=None,
        target_user_id=None
    ):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"❌ 数据库不存在: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        if target_user_id:
            cursor.execute("SELECT content FROM conversations WHERE user_id = ?", (target_user_id,))
        else:
            cursor.execute("SELECT content FROM conversations")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"❌ 读取数据库失败: {e}")
        return
    finally:
        conn.close()
    
    if not rows:
        print(f"⚠️ 没有找到用户 {target_user_id} 的聊天记录")
        return

    daily_txt = defaultdict(list)
    daily_json = defaultdict(list)

    def extract_timestamp(text):
        if not text: return None
        match = re.search(r'Current datetime:\s*([0-9:\-\sT]+)\s*\(.*?\)', text)
        if match:
            return match.group(1).strip()
        match = re.search(r'triggered at\s*([0-9:\-\sT\.\+]+)', text)
        if match:
            return match.group(1).strip()
        return None

    def extract_nickname(text):
        if not text: return None
        match = re.search(r'<system_reminder>.*?Nickname:\s*([^,\n\s]+)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def is_metadata_block(text):
        return "<system_reminder>" in text

    # 错误关键词列表（按长度降序，优先匹配更具体的）
    error_keywords = [
        "permission denied",
        "cannot open directory",
        "unable to",
        "command not found",
        "no such file",
        "is a directory",
        "not a directory",
        "invalid",
        "exception",
        "traceback",
        "error",
        "fail",
        "denied"
    ]
    error_re = re.compile('|'.join(re.escape(kw) for kw in error_keywords), re.IGNORECASE)

    for row in rows:
        row_content = decode_json_unicode(row['content'])
        if not isinstance(row_content, list): continue
        
        initial_timestamp = None
        for turn in row_content:
            cs = turn.get("content")
            if isinstance(cs, list):
                for b in cs:
                    t = b.get("text", "") if isinstance(b, dict) else str(b)
                    initial_timestamp = extract_timestamp(t)
                    if initial_timestamp: break
            elif isinstance(cs, str):
                initial_timestamp = extract_timestamp(cs)
            if initial_timestamp: break
        
        timestamp = initial_timestamp
        for turn in row_content:
            role = turn.get("role")
            contents = turn.get("content")
            tool_calls = turn.get("tool_calls")
            
            if not role: continue
            if contents is None and not tool_calls: continue

            text_messages = []
            turn_nickname = None

            def process_content_item(item):
                nonlocal timestamp, turn_nickname
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict) and item.get("type") in ["text", "plain"]:
                    text = item.get("text", "")
                else:
                    return
                ts = extract_timestamp(text)
                if ts:
                    timestamp = ts
                nick = extract_nickname(text)
                if nick:
                    turn_nickname = nick
                if not is_metadata_block(text):
                    marker = "I finished this job, here is the result:"
                    if marker in text:
                        text = text.split(marker, 1)[1]
                    text_messages.append(text.strip())

            def process_tool_calls_for_actions(calls):
                if not isinstance(calls, list):
                    return []
                actions = []
                for call in calls:
                    func = call.get("function", {})
                    name = func.get("name", "unknown")
                    args_str = func.get("arguments", "{}")
                    # 解码 Unicode 转义序列（如 \u83b7 -> 获）
                    args_str = decode_unicode_escapes(args_str)
                    # 截断过长的参数
                    if len(args_str) > 200:
                        args_preview = args_str[:200] + "..."
                    else:
                        args_preview = args_str
                    action_desc = f"[行动] 调用函数: {name}({args_preview})"
                    actions.append((call.get("id"), action_desc))
                return actions

            if isinstance(contents, list):
                for block in contents:
                    process_content_item(block)
            elif isinstance(contents, str):
                process_content_item(contents)
            
            final_text = "\n".join(text_messages).strip()
            date_key = get_date_key(timestamp) if timestamp else "unknown_date"
            if target_date and date_key != target_date:
                continue

            # assistant 普通文本
            if role == "assistant" and final_text:
                if len(final_text) <= max_assistant_chars:
                    daily_txt[date_key].append(f"[{timestamp}] {ai_name}: {final_text}")
                    daily_json[date_key].append({
                        "timestamp": timestamp,
                        "role": ai_name,
                        "content": final_text
                    })
            
            # assistant 工具调用（行动记录）
            if role == "assistant" and tool_calls:
                actions_with_id = process_tool_calls_for_actions(tool_calls)
                for tool_call_id, action_desc in actions_with_id:
                    if len(action_desc) <= max_assistant_chars:
                        daily_txt[date_key].append(f"[{timestamp}] {ai_name}: {action_desc}")
                        daily_json[date_key].append({
                            "timestamp": timestamp,
                            "role": ai_name,
                            "content": action_desc
                        })
            
            # tool 消息：只提取错误关键词
            if role == "tool":
                tool_content = ""
                if isinstance(contents, str):
                    tool_content = contents
                elif isinstance(contents, list):
                    parts = []
                    for block in contents:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    tool_content = "\n".join(parts)
                else:
                    tool_content = str(contents) if contents else ""
                
                if tool_content:
                    match = error_re.search(tool_content)
                    if match:
                        keyword = match.group(0).strip()
                        error_msg = f"[行动失败] {keyword}"
                        daily_txt[date_key].append(f"[{timestamp}] {ai_name}: {error_msg}")
                        daily_json[date_key].append({
                            "timestamp": timestamp,
                            "role": ai_name,
                            "content": error_msg
                        })
                continue

            # user 消息
            if role == "user" and final_text:
                current_username = turn_nickname if turn_nickname else username
                if len(final_text) <= max_user_chars:
                    daily_txt[date_key].append(f"[{timestamp}] {current_username}: {final_text}")
                    daily_json[date_key].append({
                        "timestamp": timestamp,
                        "role": current_username,
                        "content": final_text
                    })

    # 输出文件
    for date_key, messages in daily_json.items():
        if not messages:
            continue
        
        safe_user_id = target_user_id.replace(":", "_") if target_user_id else ""
        output_txt = output_dir / f"{date_key}_dialog_{safe_user_id}.txt"
        output_json = output_dir / f"{date_key}_dialog_{safe_user_id}.json"

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(daily_txt[date_key]))

        metadata = {
            "date": date_key,
            "total_messages": len(daily_json[date_key]),
            "platform": platform,
            "created_at": datetime.now().isoformat(timespec="seconds")
        }

        json_output = {
            "metadata": metadata,
            "conversations": daily_json[date_key]
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)

        print(f"📄 已输出：{output_json}")