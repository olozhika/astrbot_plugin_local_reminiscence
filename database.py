import re
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
                    reflection TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 数据库迁移：检查新列是否存在
            cursor.execute("PRAGMA table_info(events)")
            columns = [column[1] for column in cursor.fetchall()]
            if "reflection" not in columns:
                cursor.execute("ALTER TABLE events ADD COLUMN reflection TEXT")
            if "reinforcement_count" not in columns:
                cursor.execute(
                    "ALTER TABLE events ADD COLUMN reinforcement_count INTEGER DEFAULT 0"
                )

            cursor.execute("PRAGMA table_info(nodes)")
            node_columns = [column[1] for column in cursor.fetchall()]
            if "aliases" not in node_columns:
                cursor.execute("ALTER TABLE nodes ADD COLUMN aliases TEXT DEFAULT '[]'")
            if "related_event_ids" not in node_columns:
                cursor.execute(
                    "ALTER TABLE nodes ADD COLUMN related_event_ids TEXT DEFAULT '[]'"
                )
            if "related_theme_ids" not in node_columns:
                cursor.execute(
                    "ALTER TABLE nodes ADD COLUMN related_theme_ids TEXT DEFAULT '[]'"
                )

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
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_tags_tag_id ON event_tags(tag_id)"
            )

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

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_relations (
                    source_event_id TEXT,
                    target_event_id TEXT,
                    relation_type TEXT,  -- 'caused_by', 'related_to', 'context_for', 'conclusion_of'
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_event_id, target_event_id, relation_type),
                    FOREIGN KEY(source_event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                    FOREIGN KEY(target_event_id) REFERENCES events(event_id) ON DELETE CASCADE
                )
            """)

            # 记忆固化相关表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS thematic_memories (
                    theme_id TEXT PRIMARY KEY,
                    summary TEXT,
                    centroid_vector BLOB,
                    keywords TEXT,
                    event_count INTEGER DEFAULT 0,
                    last_summarized_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_theme_map (
                    event_id TEXT,
                    theme_id TEXT,
                    PRIMARY KEY (event_id, theme_id),
                    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                    FOREIGN KEY(theme_id) REFERENCES thematic_memories(theme_id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_config (
                    key TEXT PRIMARY KEY,
                    value TEXT
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
        """更新记忆节点。aliases、related_event_ids 和 related_theme_ids 采用追加合并策略。"""
        if not nodes:
            return
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for node in nodes:
                existing = cursor.execute(
                    "SELECT * FROM nodes WHERE name = ?", (node.name,)
                ).fetchone()
                if existing:
                    existing_aliases = set(json.loads(existing["aliases"] or "[]"))
                    new_aliases = set(node.aliases or [])
                    merged_aliases = list(existing_aliases | new_aliases)

                    existing_events = set(
                        json.loads(existing["related_event_ids"] or "[]")
                    )
                    new_events = set(node.related_event_ids or [])
                    merged_events = list(existing_events | new_events)

                    existing_themes = set(
                        json.loads(existing["related_theme_ids"] or "[]")
                    )
                    new_themes = set(node.related_theme_ids or [])
                    merged_themes = list(existing_themes | new_themes)

                    # 仅当 description 或 type 有实质变化时才刷新 last_updated，
                    # 避免 LLM 原样抄写节点时产生无意义的时间戳更新。
                    has_substantive_change = (existing["description"] or "") != (
                        node.description or ""
                    ) or (existing["type"] or "") != (node.type or "")

                    cursor.execute(
                        """
                        UPDATE nodes SET
                            type = ?,
                            description = ?,
                            aliases = ?,
                            related_event_ids = ?,
                            related_theme_ids = ?,
                            last_updated = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE last_updated END
                        WHERE name = ?
                    """,
                        (
                            node.type,
                            node.description,
                            json.dumps(merged_aliases, ensure_ascii=False),
                            json.dumps(merged_events, ensure_ascii=False),
                            json.dumps(merged_themes, ensure_ascii=False),
                            has_substantive_change,
                            node.name,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO nodes (name, type, description, aliases, related_event_ids, related_theme_ids, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                        (
                            node.name,
                            node.type,
                            node.description,
                            json.dumps(node.aliases, ensure_ascii=False),
                            json.dumps(node.related_event_ids, ensure_ascii=False),
                            json.dumps(node.related_theme_ids, ensure_ascii=False),
                        ),
                    )
            conn.commit()

    def delete_nodes(self, names: List[str]):
        """删除指定的记忆节点"""
        if not names:
            return
        with self._get_conn() as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(names))
            cursor.execute(f"DELETE FROM nodes WHERE name IN ({placeholders})", names)
            conn.commit()

    def insert_summary(
        self,
        summary: DailySummary,
        known_node_names: set[str] | None = None,
        ai_name: str = "",
    ):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO daily_reflections (date, reflection) VALUES (?, ?)",
                (summary.date, summary.daily_reflection),
            )

            # 插入/更新记忆节点 (合并追加 aliases、related_event_ids 和 related_theme_ids)
            if hasattr(summary, "nodes") and summary.nodes:
                for node in summary.nodes:
                    existing = cursor.execute(
                        "SELECT * FROM nodes WHERE name = ?", (node.name,)
                    ).fetchone()
                    if existing:
                        existing_aliases = set(json.loads(existing["aliases"] or "[]"))
                        new_aliases = set(node.aliases or [])
                        merged_aliases = list(existing_aliases | new_aliases)

                        existing_events = set(
                            json.loads(existing["related_event_ids"] or "[]")
                        )
                        new_events = set(node.related_event_ids or [])
                        merged_events = list(existing_events | new_events)

                        existing_themes = set(
                            json.loads(existing["related_theme_ids"] or "[]")
                        )
                        new_themes = set(node.related_theme_ids or [])
                        merged_themes = list(existing_themes | new_themes)

                        # 当 LLM 提交的节点不在"已知记忆节点背景"中时，使用追加策略避免覆盖
                        need_merge = (
                            known_node_names is not None
                            and node.name not in known_node_names
                        )
                        if need_merge:
                            _split_types = lambda s: {
                                t.strip() for t in re.split(r"[;,，/]", s) if t.strip()
                            }
                            existing_types = (
                                _split_types(existing["type"])
                                if existing["type"]
                                else set()
                            )
                            new_types = _split_types(node.type or "")
                            merged_type = "; ".join(sorted(existing_types | new_types))
                            existing_desc = existing["description"] or ""
                            new_desc = node.description or ""
                            merged_description = (
                                existing_desc + "\n" + new_desc
                                if existing_desc and new_desc
                                else (new_desc or existing_desc)
                            )
                        else:
                            merged_type = node.type
                            merged_description = node.description

                        has_substantive_change = (existing["description"] or "") != (
                            merged_description or ""
                        ) or (existing["type"] or "") != (merged_type or "")

                        cursor.execute(
                            """
                            UPDATE nodes SET
                                type = ?,
                                description = ?,
                                aliases = ?,
                                related_event_ids = ?,
                                related_theme_ids = ?,
                                last_updated = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE last_updated END
                            WHERE name = ?
                        """,
                            (
                                merged_type,
                                merged_description,
                                json.dumps(merged_aliases, ensure_ascii=False),
                                json.dumps(merged_events, ensure_ascii=False),
                                json.dumps(merged_themes, ensure_ascii=False),
                                has_substantive_change,
                                node.name,
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO nodes (name, type, description, aliases, related_event_ids, related_theme_ids, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                            (
                                node.name,
                                node.type,
                                node.description,
                                json.dumps(node.aliases, ensure_ascii=False),
                                json.dumps(node.related_event_ids, ensure_ascii=False),
                                json.dumps(node.related_theme_ids, ensure_ascii=False),
                            ),
                        )

            # 删除冗余节点
            if hasattr(summary, "deleted_nodes") and summary.deleted_nodes:
                placeholders = ",".join(["?"] * len(summary.deleted_nodes))
                cursor.execute(
                    f"DELETE FROM nodes WHERE name IN ({placeholders})",
                    summary.deleted_nodes,
                )

            for event in summary.events:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO events 
                    (event_id, date, narrative, emotion, importance, emotional_intensity, reflection)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.event_id,
                        summary.date,
                        event.narrative,
                        event.emotion,
                        event.importance,
                        event.emotional_intensity,
                        event.reflection,
                    ),
                )
                for tag_name in event.tags:
                    cursor.execute(
                        "INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,)
                    )
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
                    row = cursor.fetchone()
                    if row:
                        tag_id = row["id"]
                        cursor.execute(
                            "INSERT OR IGNORE INTO event_tags (event_id, tag_id) VALUES (?, ?)",
                            (event.event_id, tag_id),
                        )

            # 名称匹配：将节点名称和别名与事件叙述匹配，自动追加关联事件ID
            # 使用最长匹配优先消歧：当多个节点名称同时匹配同一事件时，只保留最长匹配
            # AI 节点特殊处理：仅关联有 reflection 的事件
            if (
                hasattr(summary, "nodes")
                and summary.nodes
                and hasattr(summary, "events")
                and summary.events
            ):
                # 识别 AI 节点
                ai_lower = ai_name.lower() if ai_name else ""
                ai_node_names = set()
                if ai_lower:
                    for node in summary.nodes:
                        if node.name.lower() == ai_lower:
                            ai_node_names.add(node.name)
                        elif any(a.lower() == ai_lower for a in (node.aliases or [])):
                            ai_node_names.add(node.name)

                # 为每个事件收集所有匹配的节点及其匹配长度
                ev_matches: dict[str, dict[str, int]] = {}
                for event in summary.events:
                    narrative = (event.narrative or "").lower()
                    importance = event.importance or 0
                    matches = {}
                    for node in summary.nodes:
                        # AI 节点特殊匹配：仅关联重要性 >= 8 的事件
                        if node.name in ai_node_names:
                            if importance >= 8:
                                matches[node.name] = len(node.name)
                            continue
                        if node.name and node.name.lower() in narrative:
                            matches[node.name] = max(
                                matches.get(node.name, 0), len(node.name)
                            )
                        for alias in node.aliases or []:
                            if alias and alias.lower() in narrative:
                                matches[node.name] = max(
                                    matches.get(node.name, 0), len(alias)
                                )
                    if matches:
                        ev_matches[event.event_id] = matches

                # 消歧：仅排除名称为另一匹配名称子串的短节点
                node_event_map: dict[str, set[str]] = {
                    n.name: set(n.related_event_ids or []) for n in summary.nodes
                }
                for ev_id, matches in ev_matches.items():
                    matched_names = list(matches.keys())
                    excluded = set()
                    for i, name_a in enumerate(matched_names):
                        for j, name_b in enumerate(matched_names):
                            if (
                                i != j
                                and name_a in name_b
                                and len(name_a) < len(name_b)
                            ):
                                excluded.add(name_a)
                    for node_name in matched_names:
                        if node_name not in excluded:
                            node_event_map[node_name].add(ev_id)

                for node in summary.nodes:
                    current_events = node_event_map.get(node.name, set())
                    if current_events:
                        new_serialized = json.dumps(
                            sorted(current_events), ensure_ascii=False
                        )
                        cursor.execute(
                            "UPDATE nodes SET related_event_ids = ? WHERE name = ?",
                            (new_serialized, node.name),
                        )

            # 插入事件关系
            if hasattr(summary, "relations") and summary.relations:
                for rel in summary.relations:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO event_relations (source_event_id, target_event_id, relation_type, confidence)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            rel.source_id,
                            rel.target_id,
                            rel.relation_type,
                            rel.confidence,
                        ),
                    )

            conn.commit()

    def delete_summary_by_date(
        self, date: str
    ) -> tuple[List[str], List[str], List[str]]:
        """
        清除指定日期的所有事件、日总结以及其与标签、主题和关系的连接，并返回：
        - 被删除的所有事件的ID (deleted_event_ids)
        - 仍保留但需要更新重心的主题ID (themes_to_update)
        - 应该被删除的主题ID (themes_to_delete)
        """
        deleted_event_ids = []
        themes_to_update = []
        themes_to_delete = []

        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 1. 查找此日期已存在的所有 event_id
            cursor.execute("SELECT event_id FROM events WHERE date = ?", (date,))
            rows = cursor.fetchall()
            deleted_event_ids = [row["event_id"] for row in rows]

            if deleted_event_ids:
                placeholders = ",".join(["?"] * len(deleted_event_ids))

                # 2. 找到这些 event_id 所关联的主题 ID
                cursor.execute(
                    f"SELECT DISTINCT theme_id FROM event_theme_map WHERE event_id IN ({placeholders})",
                    deleted_event_ids,
                )
                affected_theme_ids = [row["theme_id"] for row in cursor.fetchall()]

                # 3. 删除事件标签连接
                cursor.execute(
                    f"DELETE FROM event_tags WHERE event_id IN ({placeholders})",
                    deleted_event_ids,
                )

                # 4. 删除事件主题映射
                cursor.execute(
                    f"DELETE FROM event_theme_map WHERE event_id IN ({placeholders})",
                    deleted_event_ids,
                )

                # 5. 删除事件关系
                cursor.execute(
                    f"DELETE FROM event_relations WHERE source_event_id IN ({placeholders}) OR target_event_id IN ({placeholders})",
                    deleted_event_ids + deleted_event_ids,
                )

                # 6. 删除事件本身
                cursor.execute("DELETE FROM events WHERE date = ?", (date,))

                # 7. 处理受影响的主题 (thematic_memories)
                for theme_id in affected_theme_ids:
                    cursor.execute(
                        "SELECT COUNT(*) as count FROM event_theme_map WHERE theme_id = ?",
                        (theme_id,),
                    )
                    count = cursor.fetchone()["count"]
                    if count == 0:
                        # 这是一个空主题，直接在 SQLite 中物理删除
                        cursor.execute(
                            "DELETE FROM thematic_memories WHERE theme_id = ?",
                            (theme_id,),
                        )
                        themes_to_delete.append(theme_id)
                    else:
                        cursor.execute(
                            """
                            UPDATE thematic_memories 
                            SET event_count = ?, 
                                last_summarized_count = MIN(last_summarized_count, ?)
                            WHERE theme_id = ?
                        """,
                            (count, count, theme_id),
                        )
                        themes_to_update.append(theme_id)

            # 8. 删除每日总结 (daily_reflections)
            cursor.execute("DELETE FROM daily_reflections WHERE date = ?", (date,))

            conn.commit()

        return deleted_event_ids, themes_to_update, themes_to_delete

    def insert_relations(self, relations: List):
        """插入事件关系列表"""
        if not relations:
            return
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for rel in relations:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO event_relations (source_event_id, target_event_id, relation_type, confidence)
                    VALUES (?, ?, ?, ?)
                """,
                    (rel.source_id, rel.target_id, rel.relation_type, rel.confidence),
                )
            conn.commit()

    def get_summaries(self, days: int) -> List[dict]:
        """获取最近 days 天的完整总结（包含事件）"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 获取最近的有记录的日期
            cursor.execute(
                """
                SELECT DISTINCT date FROM daily_reflections 
                ORDER BY date DESC LIMIT ?
            """,
                (days,),
            )
            dates = [row["date"] for row in cursor.fetchall()]

            result = []
            for date in dates:
                cursor.execute(
                    "SELECT reflection FROM daily_reflections WHERE date = ?", (date,)
                )
                reflection_row = cursor.fetchone()

                cursor.execute(
                    """
                    SELECT e.*, GROUP_CONCAT(t.name) as tags
                    FROM events e
                    LEFT JOIN event_tags et ON e.event_id = et.event_id
                    LEFT JOIN tags t ON et.tag_id = t.id
                    WHERE e.date = ?
                    GROUP BY e.event_id
                """,
                    (date,),
                )
                events = []
                for ev_row in cursor.fetchall():
                    events.append(
                        {
                            "event_id": ev_row["event_id"],
                            "narrative": ev_row["narrative"],
                            "emotion": ev_row["emotion"],
                            "importance": ev_row["importance"],
                            "emotional_intensity": ev_row["emotional_intensity"],
                            "reflection": ev_row["reflection"]
                            if ev_row["reflection"] is not None
                            else "",
                            "tags": ev_row["tags"].split(",") if ev_row["tags"] else [],
                        }
                    )

                result.append(
                    {
                        "date": date,
                        "reflection": reflection_row["reflection"]
                        if reflection_row
                        else "",
                        "events": events,
                    }
                )
            return result

    def get_reflections(self, days: int) -> List[dict]:
        """获取最近 days 天的自由总结"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT date, reflection FROM daily_reflections 
                ORDER BY date DESC LIMIT ?
            """,
                (days,),
            )
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

    def get_events_by_ids(self, event_ids: List[str]) -> List[dict]:
        """根据多个 event_id 批量获取事件"""
        if not event_ids:
            return []
        with self._get_conn() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in event_ids)
            cursor.execute(
                f"SELECT * FROM events WHERE event_id IN ({placeholders})",
                event_ids,
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_events(self) -> List[dict]:
        """获取数据库中所有的事件记录"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY date ASC, event_id ASC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_event_count(self) -> int:
        """返回事件总数"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]

    def get_node_count(self) -> int:
        """返回节点总数"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            return cursor.fetchone()[0]

    def get_events_by_date(self, date: str) -> List[dict]:
        """获取指定日期的所有事件"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE date = ?", (date,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_important_events(self, days: int, min_score: int) -> List[dict]:
        """获取最近 n 天内，重要性 * 情感强度 >= m 的事件"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # SQLite date('now', 'localtime') 返回当前本地日期 YYYY-MM-DD
            # 考虑到用户可能在不同时区，这里使用 date('now', 'localtime') 或者直接 date('now')
            # 插件内部通常使用 datetime.now().strftime("%Y-%m-%d")，所以这里保持一致
            cursor.execute(
                """
                SELECT * FROM events 
                WHERE date >= date('now', 'localtime', ?) 
                AND (importance * emotional_intensity) >= ?
                ORDER BY date DESC, (importance * emotional_intensity) DESC
            """,
                (f"-{days} days", min_score),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_reflection_by_date(self, date: str) -> dict:
        """获取指定日期的自由心得"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT date, reflection FROM daily_reflections WHERE date = ?", (date,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_nodes_by_names(self, names: List[str]) -> List[dict]:
        """根据名称列表获取节点信息"""
        if not names:
            return []
        with self._get_conn() as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(names))
            # 使用 LOWER() 实现不区分大小写匹配
            lower_names = [n.lower() for n in names]
            cursor.execute(
                f"SELECT * FROM nodes WHERE LOWER(name) IN ({placeholders})",
                lower_names,
            )
            return [dict(row) for row in cursor.fetchall()]

    def search_nodes(
        self, query: str, limit: int = 1, include_description: bool = True
    ) -> List[dict]:
        """模糊搜索节点，匹配名称、别名或描述。

        排序优先级：
        0. 名称完全一致
        1. 别名精确匹配（作为 JSON 数组中的独立元素）
        2. 名称前缀匹配
        3. 别名前缀匹配（作为 JSON 数组中的独立元素）
        4. 名称包含
        5. 别名包含
        6. 描述包含
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            lower_query = query.lower()
            where_parts = ["LOWER(name) LIKE ?"]
            params = [f"%{lower_query}%"]

            where_parts.append("LOWER(aliases) LIKE ?")
            params.append(f"%{lower_query}%")

            if include_description:
                where_parts.append("LOWER(description) LIKE ?")
                params.append(f"%{lower_query}%")

            where_clause = " OR ".join(where_parts)

            # 防僵尸节点：排除"标记删除"的节点（描述被改成删除标记但条目仍在库中，会被误召回导致已删内容复活）
            where_clause = f"({where_clause}) AND (description IS NULL OR (description NOT LIKE '已删除%' AND description NOT LIKE '（节点已彻底移除%'))"

            # aliases 列存的是 JSON 数组，如 '["olozhika", "olO"]'。
            # 精确匹配：检查查询是否作为独立元素出现在 JSON 中（带双引号边界）。
            # 例如 aliases='["olozhika"]' + query='olozhika' → '%"olozhika"%' 匹配成功。
            alias_exact_pat = f'%"{lower_query}"%'
            alias_prefix_pat = f'%"{lower_query}%'

            # 排序：名称完全一致 > 别名精确匹配 > 名称前缀 > 别名前缀 > 名称包含 > 别名包含 > 描述包含
            params.extend(
                [
                    lower_query,
                    alias_exact_pat,
                    f"{lower_query}%",
                    alias_prefix_pat,
                    f"%{lower_query}%",
                    f"%{lower_query}%",
                    limit,
                ]
            )

            cursor.execute(
                f"""
                SELECT * FROM nodes 
                WHERE {where_clause}
                ORDER BY (
                    CASE 
                        WHEN LOWER(name) = ? THEN 0
                        WHEN LOWER(aliases) LIKE ? THEN 1
                        WHEN LOWER(name) LIKE ? THEN 2
                        WHEN LOWER(aliases) LIKE ? THEN 3
                        WHEN LOWER(name) LIKE ? THEN 4
                        WHEN LOWER(aliases) LIKE ? THEN 5
                        ELSE 6
                    END
                ), last_updated DESC
                LIMIT ?
            """,
                tuple(params),
            )
            return [dict(row) for row in cursor.fetchall()]

    def _has_relation(self, event_id1: str, event_id2: str) -> bool:
        """检查两个事件是否有记录的关系"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM event_relations WHERE (source_event_id = ? AND target_event_id = ?) OR (source_event_id = ? AND target_event_id = ?)",
                (event_id1, event_id2, event_id2, event_id1),
            )
            return cursor.fetchone() is not None

    def reinforce_memory(self, event_ids: List[str]):
        """增强被回忆事件的记忆强度"""
        if not event_ids:
            return
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for eid in event_ids:
                cursor.execute(
                    """
                    UPDATE events 
                    SET reinforcement_count = COALESCE(reinforcement_count, 0) + 1
                    WHERE event_id = ?
                """,
                    (eid,),
                )
            conn.commit()

    def get_consolidation_config(self, key: str) -> str:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM consolidation_config WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            return row["value"] if row else None

    def set_consolidation_config(self, key: str, value: str):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO consolidation_config (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
                (key, value),
            )
            conn.commit()

    def save_thematic_memory(
        self, theme_id: str, summary: str, keywords: str, event_count: int
    ):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO thematic_memories (theme_id, summary, keywords, event_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(theme_id) DO UPDATE SET
                    summary = excluded.summary,
                    keywords = excluded.keywords,
                    event_count = excluded.event_count,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (theme_id, summary, keywords, event_count),
            )
            conn.commit()

    def update_theme_summary(
        self, theme_id: str, summary: str, last_summarized_count: int
    ):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE thematic_memories 
                SET summary = ?, last_summarized_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE theme_id = ?
            """,
                (summary, last_summarized_count, theme_id),
            )
            conn.commit()

    def get_thematic_memory(self, theme_id: str) -> dict:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM thematic_memories WHERE theme_id = ?", (theme_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_theme_by_event_id(self, event_id: str) -> dict:
        """获取事件所属的主题信息"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT t.* FROM thematic_memories t
                JOIN event_theme_map m ON t.theme_id = m.theme_id
                WHERE m.event_id = ?
            """,
                (event_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_thematic_memories(self) -> List[dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM thematic_memories")
            return [dict(row) for row in cursor.fetchall()]

    def map_event_to_theme(self, event_id: str, theme_id: str):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO event_theme_map (event_id, theme_id) VALUES (?, ?)",
                (event_id, theme_id),
            )
            # 更新主题的事件计数
            cursor.execute(
                "SELECT COUNT(*) as count FROM event_theme_map WHERE theme_id = ?",
                (theme_id,),
            )
            count = cursor.fetchone()["count"]
            cursor.execute(
                "UPDATE thematic_memories SET event_count = ? WHERE theme_id = ?",
                (count, theme_id),
            )
            conn.commit()

    def get_events_by_theme(self, theme_id: str) -> List[dict]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.* FROM events e
                JOIN event_theme_map m ON e.event_id = m.event_id
                WHERE m.theme_id = ?
                ORDER BY e.date ASC
            """,
                (theme_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def clear_thematic_memories(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM thematic_memories")
            cursor.execute("DELETE FROM event_theme_map")
            conn.commit()

    def get_unthemed_events(self) -> List[dict]:
        """获取尚未归入任何主题的事件"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.* FROM events e
                LEFT JOIN event_theme_map m ON e.event_id = m.event_id
                WHERE m.theme_id IS NULL
                ORDER BY e.date ASC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def backfill_node_relations(self, ai_name: str = "") -> int:
        """全量回填：遍历所有事件和节点，通过名称/别名匹配建立 related_event_ids 关联。
        同时回填主题节点关联。返回更新的节点数。

        消歧策略：当多个节点名称（或别名）同时匹配同一个事件时，仅排除名称为另一
        匹配名称子串的短节点（如"协会"是"NJU科幻协会"的子串时排除"协会"），
        互相不是子串的节点（如"鬼畜猫"和"Minescraft"）会同时保留。

        AI 节点特殊处理：由于事件以第一人称记录（"我如何如何"），AI 名字不会出现在
        事件叙述中。对于 AI 名字对应的节点，仅关联 reflection 非空的事件（即有深度
        感想的事件，代表与 AI 认知成长相关的记忆）。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes")
            all_nodes = [dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT event_id, narrative, importance FROM events")
            all_events = [dict(row) for row in cursor.fetchall()]

            # 识别 AI 节点（名称或别名匹配 ai_name 的节点）
            ai_node_names = set()
            if ai_name:
                ai_lower = ai_name.lower()
                for node in all_nodes:
                    if node["name"].lower() == ai_lower:
                        ai_node_names.add(node["name"])
                        continue
                    aliases = json.loads(node.get("aliases") or "[]")
                    if any(a.lower() == ai_lower for a in aliases):
                        ai_node_names.add(node["name"])

            # 第一轮：为每个事件收集所有匹配的节点名称，记录最长匹配
            # event_id -> { matched_node_name: max_match_len }
            event_match_map: dict[str, dict[str, int]] = {}
            for ev in all_events:
                ev_id = ev["event_id"]
                narrative_lower = (ev["narrative"] or "").lower()
                importance = ev.get("importance") or 0
                matches = {}
                for node in all_nodes:
                    name = node["name"]
                    # AI 节点特殊匹配：仅关联重要性 >= 8 的事件
                    if name in ai_node_names:
                        if importance >= 8:
                            matches[name] = len(name)
                        continue
                    aliases = json.loads(node.get("aliases") or "[]")
                    # 检查主名称
                    if name and name.lower() in narrative_lower:
                        match_len = len(name)
                        if name not in matches or match_len > matches[name]:
                            matches[name] = match_len
                    # 检查别名
                    for alias in aliases:
                        if alias and alias.lower() in narrative_lower:
                            match_len = len(alias)
                            if name not in matches or match_len > matches[name]:
                                matches[name] = match_len
                if matches:
                    event_match_map[ev_id] = matches

            # 第二轮：消歧 —— 仅排除名称为另一匹配名称子串的短节点
            # 例："协会"是"NJU科幻协会"的子串 → 排除"协会"
            # 例："鬼畜猫"不是"Minescraft"的子串 → 两者都保留
            # node_name -> set of event_ids
            node_events: dict[str, set[str]] = {n["name"]: set() for n in all_nodes}
            for ev_id, matches in event_match_map.items():
                if not matches:
                    continue
                matched_names = list(matches.keys())
                excluded = set()
                for i, name_a in enumerate(matched_names):
                    for j, name_b in enumerate(matched_names):
                        if i != j and name_a in name_b and len(name_a) < len(name_b):
                            excluded.add(name_a)
                for node_name in matched_names:
                    if node_name not in excluded:
                        node_events[node_name].add(ev_id)

            # 第三轮：写入数据库（仅更新有变化的节点）
            updated_count = 0
            for node in all_nodes:
                name = node["name"]
                existing_ids = set(json.loads(node.get("related_event_ids") or "[]"))
                new_ids = node_events.get(name, set())
                merged = existing_ids | new_ids
                if merged != existing_ids:
                    cursor.execute(
                        "UPDATE nodes SET related_event_ids = ? WHERE name = ?",
                        (
                            json.dumps(sorted(merged), ensure_ascii=False),
                            name,
                        ),
                    )
                    updated_count += 1

            conn.commit()

        # 同时回填主题节点关联
        theme_updated = self.backfill_theme_node_relations()
        return updated_count

    def clear_theme_node_associations(self):
        """清除所有节点的 related_theme_ids，用于聚类重建前的清理。"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE nodes SET related_theme_ids = '[]'")
            conn.commit()

    def backfill_theme_node_relations(self) -> int:
        """全量回填：遍历所有主题总结和节点，通过名称/别名匹配建立 related_theme_ids 关联。
        返回更新的节点数。"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes")
            all_nodes = [dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT theme_id, summary FROM thematic_memories")
            all_themes = [dict(row) for row in cursor.fetchall()]

            updated_count = 0
            for node in all_nodes:
                name = node["name"]
                aliases = set(json.loads(node.get("aliases") or "[]"))
                matched_theme_ids = set(
                    json.loads(node.get("related_theme_ids") or "[]")
                )
                changed = False

                for theme in all_themes:
                    summary_lower = (theme["summary"] or "").lower()
                    if name and name.lower() in summary_lower:
                        if theme["theme_id"] not in matched_theme_ids:
                            matched_theme_ids.add(theme["theme_id"])
                            changed = True
                    for alias in aliases:
                        if alias and alias.lower() in summary_lower:
                            if theme["theme_id"] not in matched_theme_ids:
                                matched_theme_ids.add(theme["theme_id"])
                                changed = True

                if changed:
                    cursor.execute(
                        "UPDATE nodes SET related_theme_ids = ? WHERE name = ?",
                        (
                            json.dumps(sorted(matched_theme_ids), ensure_ascii=False),
                            name,
                        ),
                    )
                    updated_count += 1

            conn.commit()
            return updated_count

    def has_any_node_relations(self) -> bool:
        """检查是否已有任何节点建立了事件关联（用于判断是否需要启动时自动回填）。"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM nodes WHERE related_event_ids IS NOT NULL AND related_event_ids != '[]' LIMIT 1"
            )
            return cursor.fetchone() is not None

    def get_first_level_connected_nodes(self, node_name: str) -> List[dict]:
        """从指定节点出发，找所有一级连接节点（通过共享事件直接相连）。

        用于上下文感知的节点消歧：当在群聊中搜索时，优先返回与该群聊有
        共享事件的节点，避免跨群聊的同名别名污染。
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 1. 找到源节点的 related_event_ids
            cursor.execute(
                "SELECT related_event_ids FROM nodes WHERE name = ?", (node_name,)
            )
            row = cursor.fetchone()
            if not row or not row["related_event_ids"]:
                return []

            event_ids = set(json.loads(row["related_event_ids"]))
            if not event_ids:
                return []

            # 2. 取所有节点，找出与源节点共享事件的节点
            cursor.execute("SELECT * FROM nodes WHERE name != ?", (node_name,))
            all_nodes = [dict(r) for r in cursor.fetchall()]

            connected = []
            for node in all_nodes:
                node_event_ids = set(json.loads(node.get("related_event_ids") or "[]"))
                if node_event_ids & event_ids:  # 有交集 = 共享事件
                    connected.append(node)

            return connected
