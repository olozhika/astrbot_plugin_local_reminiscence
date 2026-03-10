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
        max_assistant_chars=1000,
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
        match = re.search(r'Current datetime:\s*([0-9:\-\sT]+)\s*\(.*?\)', text)
        return match.group(1).strip() if match else None

    def extract_nickname(text):
        """从 <system_reminder> 中提取 Nickname"""
        match = re.search(r'<system_reminder>.*?Nickname:\s*([^,\n\s]+)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    for row in rows:
        row_content = decode_json_unicode(row['content'])
        if not isinstance(row_content, list): continue
        
        timestamp = None
        for turn in row_content:
            role = turn.get("role")
            contents = turn.get("content")
            if not role or not contents: continue

            text_messages = []
            turn_nickname = None
            if isinstance(contents, list):
                for block in contents:
                    if isinstance(block, str):
                        text_messages.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        ts = extract_timestamp(text)
                        if ts:
                            timestamp = ts
                        
                        nick = extract_nickname(text)
                        if nick:
                            turn_nickname = nick
                        
                        if ts or nick:
                            continue
                        text_messages.append(text)
            elif isinstance(contents, str):
                text_messages.append(contents)

            final_text = " ".join(text_messages).strip()
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
        
        suffix = f"_{target_user_id}" if target_user_id else ""
        output_txt = output_dir / f"{date_key}_dialog{suffix}.txt"
        output_json = output_dir / f"{date_key}_dialog{suffix}.json"

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
