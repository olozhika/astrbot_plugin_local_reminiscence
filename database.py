import sqlite3
import json
from contextlib import contextmanager
from typing import List
from .models import DailySummary

class MemoryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    narrative TEXT NOT NULL,
                    emotion TEXT,
                    importance INTEGER CHECK(importance BETWEEN 1 AND 10),
                    emotional_intensity INTEGER CHECK(emotional_intensity BETWEEN 1 AND 10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_tags (
                    event_id TEXT,
                    tag_id INTEGER,
                    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                    PRIMARY KEY (event_id, tag_id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_tags_tag_id ON event_tags(tag_id)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_reflections (
                    date TEXT PRIMARY KEY,
                    reflection TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    name TEXT PRIMARY KEY,
                    type TEXT,
                    description TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def update_nodes(self, nodes: list):
        """更新记忆节点"""
        if not nodes:
            return
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for node in nodes:
                cursor.execute("""
                    INSERT INTO nodes (name, type, description, last_updated)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(name) DO UPDATE SET
                        type = excluded.type,
                        description = excluded.description,
                        last_updated = CURRENT_TIMESTAMP
                """, (node.name, node.type, node.description))
            conn.commit()

    def insert_summary(self, summary: DailySummary):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO daily_reflections (date, reflection) VALUES (?, ?)",
                (summary.date, summary.daily_reflection)
            )
            
            # 插入/更新记忆节点
            if hasattr(summary, 'nodes') and summary.nodes:
                for node in summary.nodes:
                    # 如果节点已存在，尝试合并描述（简单合并或覆盖，这里采用覆盖并记录日志）
                    cursor.execute("""
                        INSERT INTO nodes (name, type, description, last_updated)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(name) DO UPDATE SET
                            type = excluded.type,
                            description = excluded.description,
                            last_updated = CURRENT_TIMESTAMP
                    """, (node.name, node.type, node.description))

            for event in summary.events:
                cursor.execute("""
                    INSERT OR REPLACE INTO events 
                    (event_id, date, narrative, emotion, importance, emotional_intensity)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    summary.date,
                    event.narrative,
                    event.emotion,
                    event.importance,
                    event.emotional_intensity
                ))
                for tag_name in event.tags:
                    cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
                    row = cursor.fetchone()
                    if row:
                        tag_id = row['id']
                        cursor.execute(
                            "INSERT OR IGNORE INTO event_tags (event_id, tag_id) VALUES (?, ?)",
                            (event.event_id, tag_id)
                        )
            conn.commit()

    def get_summaries(self, days: int) -> List[dict]:
        """获取最近 days 天的完整总结（包含事件）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 获取最近的有记录的日期
            cursor.execute("""
                SELECT DISTINCT date FROM daily_reflections 
                ORDER BY date DESC LIMIT ?
            """, (days,))
            dates = [row['date'] for row in cursor.fetchall()]
            
            result = []
            for date in dates:
                cursor.execute("SELECT reflection FROM daily_reflections WHERE date = ?", (date,))
                reflection_row = cursor.fetchone()
                
                cursor.execute("""
                    SELECT e.*, GROUP_CONCAT(t.name) as tags
                    FROM events e
                    LEFT JOIN event_tags et ON e.event_id = et.event_id
                    LEFT JOIN tags t ON et.tag_id = t.id
                    WHERE e.date = ?
                    GROUP BY e.event_id
                """, (date,))
                events = []
                for ev_row in cursor.fetchall():
                    events.append({
                        "narrative": ev_row['narrative'],
                        "emotion": ev_row['emotion'],
                        "importance": ev_row['importance'],
                        "emotional_intensity": ev_row['emotional_intensity'],
                        "tags": ev_row['tags'].split(',') if ev_row['tags'] else []
                    })
                
                result.append({
                    "date": date,
                    "reflection": reflection_row['reflection'] if reflection_row else "",
                    "events": events
                })
            return result

    def get_reflections(self, days: int) -> List[dict]:
        """获取最近 days 天的自由总结"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, reflection FROM daily_reflections 
                ORDER BY date DESC LIMIT ?
            """, (days,))
            return [dict(row) for row in cursor.fetchall()]

    def get_event_by_id(self, event_id: str) -> dict:
        """根据 event_id 获取事件全部信息"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_events_by_date(self, date: str) -> List[dict]:
        """获取指定日期的所有事件"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE date = ?", (date,))
            return [dict(row) for row in cursor.fetchall()]

    def get_nodes_by_names(self, names: List[str]) -> List[dict]:
        """根据名称列表获取节点信息"""
        if not names:
            return []
        with self._get_conn() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(names))
            cursor.execute(f"SELECT * FROM nodes WHERE name IN ({placeholders})", names)
            return [dict(row) for row in cursor.fetchall()]

    def search_node_by_name(self, name: str) -> dict | None:
        """精确或模糊搜索单个节点"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 先尝试精确匹配
            cursor.execute("SELECT * FROM nodes WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            
            # 再尝试模糊匹配
            cursor.execute("SELECT * FROM nodes WHERE name LIKE ?", (f"%{name}%",))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
