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
    """
    提取聊天记录并按天保存为 TXT 和 JSON
    :param target_date: 如果指定，则只生成该日期的文件 (YYYY-MM-DD)
    """
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
        # 匹配 <system_reminder> 中的格式: Current datetime: 2026-03-12 17:36 (CST)
        match = re.search(r'Current datetime:\s*([0-9:\-\sT]+)\s*\(.*?\)', text)
        if match:
            return match.group(1).strip()
        # 匹配 CronJob 中的格式: triggered at 2026-03-12T09:30:00.003831+00:00
        match = re.search(r'triggered at\s*([0-9:\-\sT\.\+]+)', text)
        if match:
            return match.group(1).strip()
        return None

    def extract_nickname(text):
        """从 <system_reminder> 中提取 Nickname"""
        if not text: return None
        match = re.search(r'<system_reminder>.*?Nickname:\s*([^,\n\s]+)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def is_metadata_block(text):
        """判断是否为纯元数据块（如 system_reminder）"""
        return "<system_reminder>" in text

    for row in rows:
        row_content = decode_json_unicode(row['content'])
        if not isinstance(row_content, list): continue
        
        # 预检：为这一行（一个会话上下文）寻找第一个可用的时间戳作为初始值
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
            # 如果 content 为空且没有 tool_calls，则跳过
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
                    # 处理 CronJob 文本，只保留结果部分
                    marker = "I finished this job, here is the result:"
                    if marker in text:
                        text = text.split(marker, 1)[1]
                    text_messages.append(text.strip())

            def process_tool_calls(calls):
                if not isinstance(calls, list): return
                for call in calls:
                    func = call.get("function", {})
                    if func.get("name") == "send_message_to_user":
                        args_str = func.get("arguments", "{}")
                        try:
                            # 解析工具调用的参数字符串，json.loads 会处理 unicode 转义
                            args = json.loads(args_str)
                            msgs = args.get("messages", [])
                            if isinstance(msgs, list):
                                for m in msgs:
                                    process_content_item(m)
                        except Exception:
                            pass

            if isinstance(contents, list):
                for block in contents:
                    process_content_item(block)
            elif isinstance(contents, str):
                process_content_item(contents)
            
            if tool_calls:
                process_tool_calls(tool_calls)

            final_text = "\n".join(text_messages).strip()
            if not final_text: continue

            date_key = get_date_key(timestamp) if timestamp else "unknown_date"
            
            # 如果指定了目标日期，且当前日期不匹配，则跳过
            if target_date and date_key != target_date:
                continue

            current_username = turn_nickname if turn_nickname else username

            if role == "user":
                if len(final_text) <= max_user_chars:
                    daily_txt[date_key].append(f"[{timestamp}] {current_username}: {final_text}")
                    daily_json[date_key].append({
                        "timestamp": timestamp,
                        "role": current_username,
                        "content": final_text
                    })
            elif role == "assistant":
                if len(final_text) <= max_assistant_chars:
                    daily_txt[date_key].append(f"[{timestamp}] {ai_name}: {final_text}")
                    daily_json[date_key].append({
                        "timestamp": timestamp,
                        "role": ai_name,
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
