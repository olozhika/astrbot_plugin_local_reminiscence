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

    for row in rows:
        row_content = decode_json_unicode(row['content'])
        if not isinstance(row_content, list): continue
        
        timestamp = None
        for turn in row_content:
            role = turn.get("role")
            contents = turn.get("content")
            if not role or not contents: continue

            text_messages = []
            if isinstance(contents, list):
                for block in contents:
                    if isinstance(block, str):
                        text_messages.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        ts = extract_timestamp(text)
                        if ts:
                            timestamp = ts
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

            if role == "user":
                if len(final_text) <= max_user_chars:
                    daily_txt[date_key].append(f"[{timestamp}] {username}: {final_text}")
                    daily_json[date_key].append({
                        "timestamp": timestamp,
                        "role": username,
                        "content": final_text
                    })
            elif role == "assistant":
                if len(final_text) <= max_assistant_chars:
                    daily_txt[date_key].append(f"[{timestamp}] Lanya: {final_text}")
                    daily_json[date_key].append({
                        "timestamp": timestamp,
                        "role": "Lanya",
                        "content": final_text
                    })

    # 输出文件
    for date_key, messages in daily_json.items():
        if not messages:
            continue
        output_txt = output_dir / f"{date_key}_dialog.txt"
        output_json = output_dir / f"{date_key}_dialog.json"

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
import re
import os
import json
import sqlite3
import pandas as pd
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# --- Load Config from _conf_schema.json ---
def load_config():
    # 从 _conf_schema.json 加载
    schema_path = Path(__file__).parent / '_conf_schema.json'
    if schema_path.exists():
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
            return {k: v.get('default') if isinstance(v, dict) else v for k, v in schema.items()}
    return {}

CONFIG = load_config()
CORE_ADDRESS = CONFIG.get("core_address", "")
TARGET_USER_ID = CONFIG.get("target_user_id", "Lanya_QQ:FriendMessage:947628188")
DEFAULT_OUTPUT_DIR = os.path.join(CORE_ADDRESS, CONFIG.get("dialog_folder", "APLR_chat_history"))

def decode_json_unicode(s):
    try:
        return json.loads(s)
    except:
        return s
    
def get_date_key(ts):
    if ts is None:
        return "unknown_date"
    ts_norm = ts.replace(" ", "T")
    try:
        dt = datetime.fromisoformat(ts_norm)
        return dt.date().isoformat()
    except:
        return ts.split(" ")[0].split("T")[0]

def clean_dialogue_with_different_limits(
        username="olozhika",
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
    db_path = Path(CORE_ADDRESS) / "data" / "data_v4.db"
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"❌ 数据库不存在: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    try:
        # 如果指定了日期，可以尝试在 SQL 层面优化，但由于 content 是 JSON 且日期在 JSON 内部，
        # 这里依然读取全表，但在处理时过滤。
        df = pd.read_sql_query("SELECT * FROM conversations", conn)
    except Exception as e:
        print(f"❌ 读取数据库失败: {e}")
        return
    finally:
        conn.close()
    
    # 过滤目标用户
    if target_user_id:
        df = df[df['user_id'] == target_user_id]
    
    if df.empty:
        print(f"⚠️ 没有找到用户 {target_user_id} 的聊天记录")
        return

    data = df['content'].apply(decode_json_unicode)

    daily_txt = defaultdict(list)
    daily_json = defaultdict(list)

    def extract_timestamp(text):
        match = re.search(r'Current datetime:\s*([0-9:\-\sT]+)\s*\(.*?\)', text)
        return match.group(1).strip() if match else None

    for index, row_content in data.items():
        if not isinstance(row_content, list): continue
        
        timestamp = None
        for turn in row_content:
            role = turn.get("role")
            contents = turn.get("content")
            if not role or not contents: continue

            text_messages = []
            if isinstance(contents, list):
                for block in contents:
                    if isinstance(block, str):
                        text_messages.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        ts = extract_timestamp(text)
                        if ts:
                            timestamp = ts
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

            if role == "user":
                if len(final_text) <= max_user_chars:
                    daily_txt[date_key].append(f"[{timestamp}] {username}: {final_text}")
                    daily_json[date_key].append({
                        "timestamp": timestamp,
                        "role": username,
                        "content": final_text
                    })
            elif role == "assistant":
                if len(final_text) <= max_assistant_chars:
                    daily_txt[date_key].append(f"[{timestamp}] Lanya: {final_text}")
                    daily_json[date_key].append({
                        "timestamp": timestamp,
                        "role": "Lanya",
                        "content": final_text
                    })

    # 输出文件
    for date_key, messages in daily_json.items():
        if not messages:
            continue
        output_txt = output_dir / f"{date_key}_dialog.txt"
        output_json = output_dir / f"{date_key}_dialog.json"

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

if __name__ == "__main__":
    # 支持从命令行指定日期、用户 ID 和用户名
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    target_user_id = sys.argv[2] if len(sys.argv) > 2 else TARGET_USER_ID
    username = sys.argv[3] if len(sys.argv) > 3 else "olozhika"
    
    clean_dialogue_with_different_limits(
        username=username,
        platform="QQ",
        target_date=target_date,
        target_user_id=target_user_id
    )
