from __future__ import annotations

import json
import logging
import os
import sys
import time
import hmac
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import psycopg
from flask import Blueprint, Flask, jsonify, request
from openai import OpenAI
from psycopg.rows import dict_row

# =========================
# logging
# =========================

def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logger = get_logger(__name__)

# =========================
# config
# =========================

@dataclass
class Settings:
    app_env: str
    log_level: str
    database_url: str
    tg_bot_token: str
    openai_api_key: str
    llm_model_name: str
    default_timezone: str
    webhook_host: str
    webhook_port: int

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            app_env=os.getenv("APP_ENV", "dev"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            database_url=os.getenv("DATABASE_URL", "").strip(),
            tg_bot_token=os.getenv("TG_BOT_TOKEN", "").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-5.4"),
            default_timezone=os.getenv("DEFAULT_TIMEZONE", "UTC"),
            webhook_host=os.getenv("WEBHOOK_HOST", "0.0.0.0"),
            webhook_port=int(os.getenv("WEBHOOK_PORT", "8080")),
        )

    def validate(self) -> None:
        missing: list[str] = []
        if not self.database_url:
            missing.append("DATABASE_URL")
        if not self.tg_bot_token:
            missing.append("TG_BOT_TOKEN")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


# =========================
# time utils
# =========================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_in_timezone(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def is_silence_active(silence_until: datetime | None) -> bool:
    return bool(silence_until and silence_until > utc_now())


# =========================
# schemas
# =========================

@dataclass
class InboundMessage:
    business_account_id: int
    tg_business_account_id: str
    user_id: int
    tg_user_id: str
    conversation_id: int | None
    sender_type: str
    message_type: str
    text: str | None
    media_url: str | None
    raw_payload: dict[str, Any]
    sent_at: datetime


@dataclass
class ConversationContext:
    conversation_id: int
    business_account_id: int
    user_id: int
    current_stage: str | None = None
    current_chat_mode: str | None = None
    current_mainline: str | None = None
    ai_enabled: bool = True
    manual_takeover_status: str = "inactive"
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    recent_summary: str = ""
    long_term_memory: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserStateSnapshot:
    ops_category: str = "new_user"
    project_id: int | None = None
    project_segment_id: int | None = None
    tags: list[str] = field(default_factory=list)
    relationship_score: float = 0.0
    trust_score: float = 0.0
    comfort_score: float = 0.0
    current_heat: float = 0.0
    marketing_tolerance: float = 0.0


@dataclass
class UnderstandingResult:
    user_intent: str
    need_type: str
    emotion_state: str | None
    boundary_signal: str | None
    resistance_signal: str | None
    product_interest_signal: bool
    explicit_product_query: bool
    high_intent_signal: bool
    current_mainline_should_continue: str
    recommended_chat_mode: str
    recommended_goal: str
    reason: str


@dataclass
class StageDecision:
    stage: str
    changed: bool
    reason: str


@dataclass
class ModeDecision:
    chat_mode: str
    changed: bool
    reason: str


@dataclass
class ProjectDecision:
    project_id: int | None
    candidate_projects: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "none"
    changed: bool = False
    reason: str = ""


@dataclass
class IntentDecision:
    score: float
    level: str
    reason: str


@dataclass
class TagDecision:
    add_tags: list[str] = field(default_factory=list)
    remove_tags: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ReplyPlan:
    should_reply: bool = True
    should_answer_question: bool = True
    should_continue_product: bool = False
    should_self_share: bool = False
    should_send_material: bool = False
    should_leave_space: bool = False
    should_prepare_human_escalation: bool = False
    goal: str = "keep_replying"
    reason: str = ""


@dataclass
class FinalReply:
    text: str
    delay_seconds: int = 0
    used_material_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumeDecision:
    should_resume: bool
    mode: str
    goal: str
    opening_hint: str
    reason: str


# =========================
# database
# =========================

class Database:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.conn: psycopg.Connection | None = None

    def connect(self) -> None:
        if self.conn is None:
            self.conn = psycopg.connect(self.dsn, autocommit=False, row_factory=dict_row)

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    @contextmanager
    def cursor(self):
        if self.conn is None:
            raise RuntimeError("Database connection is not initialized.")
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    @contextmanager
    def transaction(self):
        if self.conn is None:
            raise RuntimeError("Database connection is not initialized.")
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise


# =========================
# repositories
# =========================

class BusinessAccountRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_by_tg_business_account_id(self, tg_business_account_id: str) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM business_accounts WHERE tg_business_account_id=%s LIMIT 1", (tg_business_account_id,))
            return cur.fetchone()

    def create_if_not_exists(self, tg_business_account_id: str, display_name: str, username: str | None = None) -> dict:
        existing = self.get_by_tg_business_account_id(tg_business_account_id)
        if existing:
            return existing
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO business_accounts (tg_business_account_id, display_name, username)
                    VALUES (%s, %s, %s)
                    RETURNING *
                    """,
                    (tg_business_account_id, display_name, username),
                )
                return cur.fetchone()


class BootstrapRepository:
    DEFAULT_TAGS: list[tuple[str, str, str]] = [
        ("high_intent", "intent", "short_term"),
        ("recently_busy", "status", "short_term"),
        ("cautious", "risk", "long_term"),
        ("product_interest", "intent", "short_term"),
        ("followup_worthy", "status", "short_term"),
        ("need_human", "status", "short_term"),
    ]

    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure_default_ai_control_settings(self, business_account_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ai_control_settings (
                        business_account_id, global_ai_enabled, new_user_ai_enabled,
                        followup_user_ai_enabled, high_intent_ai_enabled,
                        archived_user_ai_enabled, invalid_user_ai_enabled, blacklist_ai_enabled
                    )
                    VALUES (%s, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE)
                    ON CONFLICT (business_account_id) DO NOTHING
                    """,
                    (business_account_id,),
                )

    def ensure_default_persona_core(self, business_account_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO business_account_persona_core (
                        business_account_id, gender, age_feel, role, professional_vibe,
                        relationship_style, tone_base, tone_pressure, marketing_model,
                        self_sharing_enabled, daily_life_talk_enabled, project_before_tag, manual_override_priority
                    ) VALUES (%s,'female',33,'financial_wealth_planning_advisor','light_business',
                             'warm_and_low_pressure','natural','low_pressure','light_marketing',
                             TRUE,TRUE,TRUE,TRUE)
                    ON CONFLICT (business_account_id) DO NOTHING
                    """,
                    (business_account_id,),
                )

    def ensure_default_tags(self, business_account_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                for name, tag_group, tag_type in self.DEFAULT_TAGS:
                    cur.execute(
                        """
                        INSERT INTO tags (business_account_id, name, tag_group, tag_type, description, is_active)
                        VALUES (%s, %s, %s, %s, %s, TRUE)
                        ON CONFLICT (business_account_id, name) DO NOTHING
                        """,
                        (business_account_id, name, tag_group, tag_type, f"default tag: {name}"),
                    )


class ConversationRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_or_create_conversation(self, business_account_id: int, user_id: int) -> int:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT id FROM conversations WHERE business_account_id=%s AND user_id=%s LIMIT 1",
                (business_account_id, user_id),
            )
            row = cur.fetchone()
            if row:
                return int(row["id"])
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (
                        business_account_id, user_id, current_stage, current_chat_mode,
                        current_mainline, ai_enabled, manual_takeover_status, last_message_at
                    ) VALUES (%s,%s,NULL,NULL,NULL,TRUE,'inactive',NOW()) RETURNING id
                    """,
                    (business_account_id, user_id),
                )
                return int(cur.fetchone()["id"])

    def save_message(self, conversation_id: int, sender_type: str, message_type: str, text: str | None, raw_payload: dict | None = None, media_url: str | None = None) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (conversation_id, sender_type, message_type, content_text, media_url, raw_payload_json)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    (conversation_id, sender_type, message_type, text, media_url, raw_payload or {}),
                )
                cur.execute("UPDATE conversations SET last_message_at=NOW(), updated_at=NOW() WHERE id=%s", (conversation_id,))

    def get_context(self, conversation_id: int) -> ConversationContext:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT id AS conversation_id, business_account_id, user_id,
                       current_stage, current_chat_mode, current_mainline,
                       ai_enabled, manual_takeover_status
                FROM conversations WHERE id=%s LIMIT 1
                """,
                (conversation_id,),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"Conversation not found: {conversation_id}")
            cur.execute(
                """
                SELECT id, sender_type, message_type, content_text, media_url, created_at
                FROM messages WHERE conversation_id=%s ORDER BY created_at DESC LIMIT 20
                """,
                (conversation_id,),
            )
            recent_rows = list(reversed(cur.fetchall()))
            cur.execute(
                "SELECT summary_text FROM conversation_summaries WHERE conversation_id=%s ORDER BY created_at DESC LIMIT 1",
                (conversation_id,),
            )
            summary_row = cur.fetchone()
        return ConversationContext(
            conversation_id=int(row["conversation_id"]),
            business_account_id=int(row["business_account_id"]),
            user_id=int(row["user_id"]),
            current_stage=row.get("current_stage"),
            current_chat_mode=row.get("current_chat_mode"),
            current_mainline=row.get("current_mainline"),
            ai_enabled=bool(row["ai_enabled"]),
            manual_takeover_status=row.get("manual_takeover_status") or "inactive",
            recent_messages=recent_rows,
            recent_summary=(summary_row["summary_text"] if summary_row else "") or "",
            long_term_memory={},
        )

    def update_conversation_state(self, conversation_id: int, current_stage: str | None, current_chat_mode: str | None, current_mainline: str | None) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE conversations SET current_stage=%s, current_chat_mode=%s,
                    current_mainline=%s, updated_at=NOW() WHERE id=%s
                    """,
                    (current_stage, current_chat_mode, current_mainline, conversation_id),
                )

    def set_manual_takeover_status(self, conversation_id: int, status: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET manual_takeover_status=%s, updated_at=NOW() WHERE id=%s",
                    (status, conversation_id),
                )

    def set_last_ai_reply_at(self, conversation_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE conversations SET last_ai_reply_at=NOW(), updated_at=NOW() WHERE id=%s", (conversation_id,))


class UserRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_or_create_user(self, tg_user_id: str, display_name: str | None = None) -> int:
        with self.db.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE tg_user_id=%s LIMIT 1", (tg_user_id,))
            row = cur.fetchone()
            if row:
                return int(row["id"])
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (tg_user_id, display_name, first_seen_at, last_seen_at, is_blocked, created_at, updated_at)
                    VALUES (%s,%s,NOW(),NOW(),FALSE,NOW(),NOW()) RETURNING id
                    """,
                    (tg_user_id, display_name),
                )
                return int(cur.fetchone()["id"])

    def get_user_profile(self, business_account_id: int, user_id: int) -> dict:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT id, tg_user_id, username, display_name, language_code, first_seen_at, last_seen_at, is_blocked FROM users WHERE id=%s LIMIT 1",
                (user_id,),
            )
            user_row = cur.fetchone()
            cur.execute(
                "SELECT ops_category, source, is_locked, updated_at FROM user_ops_status WHERE business_account_id=%s AND user_id=%s LIMIT 1",
                (business_account_id, user_id),
            )
            ops_row = cur.fetchone()
            cur.execute(
                """
                SELECT ups.project_id, ups.status, ups.source, ups.is_locked, p.name AS project_name
                FROM user_project_state ups LEFT JOIN projects p ON ups.project_id=p.id
                WHERE ups.business_account_id=%s AND ups.user_id=%s LIMIT 1
                """,
                (business_account_id, user_id),
            )
            project_row = cur.fetchone()
            cur.execute(
                """
                SELECT upss.project_segment_id, ps.name AS segment_name
                FROM user_project_segment_state upss LEFT JOIN project_segments ps ON upss.project_segment_id=ps.id
                WHERE upss.business_account_id=%s AND upss.user_id=%s LIMIT 1
                """,
                (business_account_id, user_id),
            )
            segment_row = cur.fetchone()
            cur.execute(
                """
                SELECT t.name FROM user_tags ut JOIN tags t ON ut.tag_id=t.id
                WHERE ut.business_account_id=%s AND ut.user_id=%s AND ut.is_active=TRUE
                ORDER BY ut.created_at ASC
                """,
                (business_account_id, user_id),
            )
            tag_rows = cur.fetchall()
        return {
            "user": user_row or {},
            "ops_status": ops_row or {},
            "project_state": project_row or {},
            "segment_state": segment_row or {},
            "tags": [r["name"] for r in tag_rows],
        }

    def get_user_state_snapshot(self, business_account_id: int, user_id: int) -> UserStateSnapshot:
        with self.db.cursor() as cur:
            cur.execute("SELECT ops_category FROM user_ops_status WHERE business_account_id=%s AND user_id=%s LIMIT 1", (business_account_id, user_id))
            ops_row = cur.fetchone()
            cur.execute("SELECT project_id FROM user_project_state WHERE business_account_id=%s AND user_id=%s LIMIT 1", (business_account_id, user_id))
            project_row = cur.fetchone()
            cur.execute("SELECT project_segment_id FROM user_project_segment_state WHERE business_account_id=%s AND user_id=%s LIMIT 1", (business_account_id, user_id))
            segment_row = cur.fetchone()
            cur.execute(
                "SELECT t.name FROM user_tags ut JOIN tags t ON ut.tag_id=t.id WHERE ut.business_account_id=%s AND ut.user_id=%s AND ut.is_active=TRUE ORDER BY ut.created_at ASC",
                (business_account_id, user_id),
            )
            tag_rows = cur.fetchall()
        return UserStateSnapshot(
            ops_category=(ops_row["ops_category"] if ops_row else "new_user"),
            project_id=(project_row["project_id"] if project_row else None),
            project_segment_id=(segment_row["project_segment_id"] if segment_row else None),
            tags=[r["name"] for r in tag_rows],
        )

    def update_project_state(self, business_account_id: int, user_id: int, project_id: int | None, reason: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_project_state (
                        business_account_id, user_id, project_id, candidate_projects_json,
                        source, reason_text, confidence, status, is_locked, updated_by, created_at, updated_at
                    ) VALUES (%s,%s,%s,'[]'::jsonb,'ai',%s,0.7,'classified',FALSE,'system',NOW(),NOW())
                    ON CONFLICT (business_account_id, user_id)
                    DO UPDATE SET project_id=EXCLUDED.project_id, source=EXCLUDED.source,
                                  reason_text=EXCLUDED.reason_text, confidence=EXCLUDED.confidence,
                                  status=EXCLUDED.status, updated_by=EXCLUDED.updated_by, updated_at=NOW()
                    """,
                    (business_account_id, user_id, project_id, reason),
                )

    def apply_tag_decision(self, business_account_id: int, user_id: int, tag_decision: TagDecision) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                for tag_name in tag_decision.add_tags:
                    cur.execute("SELECT id FROM tags WHERE business_account_id=%s AND name=%s LIMIT 1", (business_account_id, tag_name))
                    tag_row = cur.fetchone()
                    if not tag_row:
                        continue
                    cur.execute(
                        """
                        INSERT INTO user_tags (
                            business_account_id, user_id, tag_id, source, reason_text,
                            confidence, is_active, is_locked, expires_at, created_by, created_at, updated_at
                        ) VALUES (%s,%s,%s,'ai',%s,0.7,TRUE,FALSE,NULL,'system',NOW(),NOW())
                        ON CONFLICT (business_account_id, user_id, tag_id)
                        DO UPDATE SET source=EXCLUDED.source, reason_text=EXCLUDED.reason_text,
                                      confidence=EXCLUDED.confidence, is_active=TRUE, updated_at=NOW()
                        """,
                        (business_account_id, user_id, tag_row["id"], tag_decision.reason),
                    )
                for tag_name in tag_decision.remove_tags:
                    cur.execute("SELECT id FROM tags WHERE business_account_id=%s AND name=%s LIMIT 1", (business_account_id, tag_name))
                    tag_row = cur.fetchone()
                    if not tag_row:
                        continue
                    cur.execute(
                        "UPDATE user_tags SET is_active=FALSE, updated_at=NOW() WHERE business_account_id=%s AND user_id=%s AND tag_id=%s",
                        (business_account_id, user_id, tag_row["id"]),
                    )

    def set_project_manual(self, business_account_id: int, user_id: int, project_id: int | None, reason_text: str, updated_by: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_project_state (
                        business_account_id, user_id, project_id, candidate_projects_json,
                        source, reason_text, confidence, status, is_locked, updated_by, created_at, updated_at
                    ) VALUES (%s,%s,%s,'[]'::jsonb,'manual',%s,1.0,'manual_confirmed',TRUE,%s,NOW(),NOW())
                    ON CONFLICT (business_account_id, user_id)
                    DO UPDATE SET project_id=EXCLUDED.project_id, source=EXCLUDED.source,
                                  reason_text=EXCLUDED.reason_text, confidence=EXCLUDED.confidence,
                                  status=EXCLUDED.status, is_locked=EXCLUDED.is_locked,
                                  updated_by=EXCLUDED.updated_by, updated_at=NOW()
                    """,
                    (business_account_id, user_id, project_id, reason_text, updated_by),
                )

    def add_manual_tag(self, business_account_id: int, user_id: int, tag_name: str, updated_by: str, reason_text: str = "") -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("SELECT id FROM tags WHERE business_account_id=%s AND name=%s LIMIT 1", (business_account_id, tag_name))
                tag_row = cur.fetchone()
                if not tag_row:
                    raise RuntimeError(f"Tag not found: {tag_name}")
                cur.execute(
                    """
                    INSERT INTO user_tags (
                        business_account_id, user_id, tag_id, source, reason_text,
                        confidence, is_active, is_locked, expires_at, created_by, created_at, updated_at
                    ) VALUES (%s,%s,%s,'manual',%s,1.0,TRUE,TRUE,NULL,%s,NOW(),NOW())
                    ON CONFLICT (business_account_id, user_id, tag_id)
                    DO UPDATE SET source=EXCLUDED.source, reason_text=EXCLUDED.reason_text,
                                  confidence=EXCLUDED.confidence, is_active=TRUE, is_locked=TRUE,
                                  created_by=EXCLUDED.created_by, updated_at=NOW()
                    """,
                    (business_account_id, user_id, tag_row["id"], reason_text, updated_by),
                )

    def remove_manual_tag(self, business_account_id: int, user_id: int, tag_name: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("SELECT id FROM tags WHERE business_account_id=%s AND name=%s LIMIT 1", (business_account_id, tag_name))
                tag_row = cur.fetchone()
                if not tag_row:
                    return
                cur.execute(
                    "UPDATE user_tags SET is_active=FALSE, updated_at=NOW() WHERE business_account_id=%s AND user_id=%s AND tag_id=%s",
                    (business_account_id, user_id, tag_row["id"]),
                )

    def set_ops_category_manual(self, business_account_id: int, user_id: int, ops_category: str, reason_text: str, updated_by: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_ops_status (
                        business_account_id, user_id, ops_category, reason_text,
                        source, is_locked, updated_by, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,'manual',TRUE,%s,NOW(),NOW())
                    ON CONFLICT (business_account_id, user_id)
                    DO UPDATE SET ops_category=EXCLUDED.ops_category, reason_text=EXCLUDED.reason_text,
                                  source=EXCLUDED.source, is_locked=EXCLUDED.is_locked,
                                  updated_by=EXCLUDED.updated_by, updated_at=NOW()
                    """,
                    (business_account_id, user_id, ops_category, reason_text, updated_by),
                )


class SettingsRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_global_ai_enabled(self, business_account_id: int) -> bool:
        with self.db.cursor() as cur:
            cur.execute("SELECT global_ai_enabled FROM ai_control_settings WHERE business_account_id=%s LIMIT 1", (business_account_id,))
            row = cur.fetchone()
        return True if not row else bool(row["global_ai_enabled"])

    def get_ops_category_ai_enabled(self, business_account_id: int, ops_category: str) -> bool:
        field_map = {
            "new_user": "new_user_ai_enabled",
            "followup_user": "followup_user_ai_enabled",
            "high_intent_user": "high_intent_ai_enabled",
            "archived_user": "archived_user_ai_enabled",
            "invalid_user": "invalid_user_ai_enabled",
            "blacklist_user": "blacklist_ai_enabled",
        }
        field_name = field_map.get(ops_category)
        if not field_name:
            return True
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {field_name} AS enabled FROM ai_control_settings WHERE business_account_id=%s LIMIT 1", (business_account_id,))
            row = cur.fetchone()
        return True if not row else bool(row["enabled"])


class UserControlRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_user_ai_control(self, conversation_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM user_ai_controls WHERE conversation_id=%s LIMIT 1", (conversation_id,))
            return cur.fetchone()

    def set_ai_override(self, conversation_id: int, ai_enabled_override: bool | None, reason_text: str, updated_by: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_ai_controls (conversation_id, ai_enabled_override, manual_takeover_forced, silence_until, reason_text, updated_by, updated_at)
                    VALUES (%s,%s,FALSE,NULL,%s,%s,NOW())
                    ON CONFLICT (conversation_id)
                    DO UPDATE SET ai_enabled_override=EXCLUDED.ai_enabled_override,
                                  reason_text=EXCLUDED.reason_text,
                                  updated_by=EXCLUDED.updated_by,
                                  updated_at=NOW()
                    """,
                    (conversation_id, ai_enabled_override, reason_text, updated_by),
                )

    def clear_ai_override(self, conversation_id: int, updated_by: str) -> None:
        self.set_ai_override(conversation_id, None, "clear ai override", updated_by)

    def set_manual_takeover(self, conversation_id: int, enabled: bool, reason_text: str, updated_by: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_ai_controls (conversation_id, ai_enabled_override, manual_takeover_forced, silence_until, reason_text, updated_by, updated_at)
                    VALUES (%s,NULL,%s,NULL,%s,%s,NOW())
                    ON CONFLICT (conversation_id)
                    DO UPDATE SET manual_takeover_forced=EXCLUDED.manual_takeover_forced,
                                  reason_text=EXCLUDED.reason_text,
                                  updated_by=EXCLUDED.updated_by,
                                  updated_at=NOW()
                    """,
                    (conversation_id, enabled, reason_text, updated_by),
                )


class MaterialRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_business_account_display_name(self, business_account_id: int) -> str:
        with self.db.cursor() as cur:
            cur.execute("SELECT display_name FROM business_accounts WHERE id=%s LIMIT 1", (business_account_id,))
            row = cur.fetchone()
        return "Business Account" if not row else row["display_name"] or "Business Account"

    def get_persona_materials(self, business_account_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM persona_materials WHERE business_account_id=%s AND is_active=TRUE ORDER BY priority ASC, id ASC",
                (business_account_id,),
            )
            return cur.fetchall()

    def get_project_materials(self, project_id: int, material_type: str | None = None) -> list[dict]:
        with self.db.cursor() as cur:
            if material_type:
                cur.execute(
                    "SELECT * FROM project_materials WHERE project_id=%s AND is_active=TRUE AND material_type=%s ORDER BY priority ASC, id ASC",
                    (project_id, material_type),
                )
            else:
                cur.execute(
                    "SELECT * FROM project_materials WHERE project_id=%s AND is_active=TRUE ORDER BY priority ASC, id ASC",
                    (project_id,),
                )
            return cur.fetchall()


class ProjectRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def list_active_projects(self, business_account_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM projects WHERE business_account_id=%s AND is_active=TRUE ORDER BY id ASC",
                (business_account_id,),
            )
            return cur.fetchall()

    def get_project_segments(self, project_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM project_segments WHERE project_id=%s AND is_active=TRUE ORDER BY sort_order ASC, id ASC",
                (project_id,),
            )
            return cur.fetchall()


class ScriptRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_project_scripts(self, project_id: int, categories: list[str] | None = None) -> list[dict]:
        with self.db.cursor() as cur:
            if categories:
                cur.execute(
                    "SELECT * FROM project_scripts WHERE project_id=%s AND is_active=TRUE AND category=ANY(%s) ORDER BY priority ASC, id ASC",
                    (project_id, categories),
                )
            else:
                cur.execute(
                    "SELECT * FROM project_scripts WHERE project_id=%s AND is_active=TRUE ORDER BY priority ASC, id ASC",
                    (project_id,),
                )
            return cur.fetchall()


class ReceiptRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create_high_intent_receipt(self, business_account_id: int, user_id: int, title: str, content_json: dict) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO receipts (
                        business_account_id, user_id, receipt_type, title, content_text, content_json, status, created_at
                    ) VALUES (%s,%s,'high_intent',%s,%s,%s,'pending',NOW())
                    """,
                    (business_account_id, user_id, title, title, content_json),
                )

    def list_recent_receipts(self, business_account_id: int, user_id: int, limit: int = 10) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM receipts WHERE business_account_id=%s AND user_id=%s ORDER BY created_at DESC LIMIT %s",
                (business_account_id, user_id, limit),
            )
            return cur.fetchall()

    def count_pending_receipts(self, business_account_id: int) -> int:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM receipts WHERE business_account_id=%s AND status='pending'",
                (business_account_id,),
            )
            row = cur.fetchone()
            return int(row["cnt"] or 0)


class AdminQueueRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_queue_item(self, business_account_id: int, user_id: int, queue_type: str, priority_score: float, reason_text: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    SELECT id FROM admin_priority_queue
                    WHERE business_account_id=%s AND user_id=%s AND queue_type=%s AND status IN ('open','processing')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (business_account_id, user_id, queue_type),
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "UPDATE admin_priority_queue SET priority_score=%s, reason_text=%s, updated_at=NOW() WHERE id=%s",
                        (priority_score, reason_text, row["id"]),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO admin_priority_queue (
                            business_account_id, user_id, queue_type, priority_score, reason_text, status, created_at, updated_at
                        ) VALUES (%s,%s,%s,%s,%s,'open',NOW(),NOW())
                        """,
                        (business_account_id, user_id, queue_type, priority_score, reason_text),
                    )

    def list_open_items(self, business_account_id: int, queue_type: str | None = None, limit: int = 50) -> list[dict]:
        with self.db.cursor() as cur:
            if queue_type:
                cur.execute(
                    "SELECT * FROM admin_priority_queue WHERE business_account_id=%s AND queue_type=%s AND status='open' ORDER BY priority_score DESC, updated_at DESC LIMIT %s",
                    (business_account_id, queue_type, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM admin_priority_queue WHERE business_account_id=%s AND status='open' ORDER BY priority_score DESC, updated_at DESC LIMIT %s",
                    (business_account_id, limit),
                )
            return cur.fetchall()

    def list_items_by_user(self, business_account_id: int, user_id: int, limit: int = 20) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM admin_priority_queue WHERE business_account_id=%s AND user_id=%s ORDER BY updated_at DESC LIMIT %s",
                (business_account_id, user_id, limit),
            )
            return cur.fetchall()

    def mark_processing(self, queue_item_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE admin_priority_queue SET status='processing', updated_at=NOW() WHERE id=%s", (queue_item_id,))


class HandoverRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create_handover_session(self, conversation_id: int, started_by: str, handover_reason: str) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "INSERT INTO handover_sessions (conversation_id, started_by, handover_reason, status, started_at) VALUES (%s,%s,%s,'active',NOW()) RETURNING *",
                    (conversation_id, started_by, handover_reason),
                )
                return cur.fetchone()

    def end_handover_session(self, session_id: int, ended_by: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE handover_sessions SET status='ended', ended_by=%s, ended_at=NOW() WHERE id=%s", (ended_by, session_id))

    def get_active_handover_session(self, conversation_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM handover_sessions WHERE conversation_id=%s AND status='active' ORDER BY started_at DESC LIMIT 1", (conversation_id,))
            return cur.fetchone()

    def get_handover_messages(self, handover_session_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute("SELECT conversation_id, started_at, ended_at FROM handover_sessions WHERE id=%s LIMIT 1", (handover_session_id,))
            session = cur.fetchone()
            if not session:
                return []
            if session["ended_at"] is None:
                cur.execute(
                    "SELECT * FROM messages WHERE conversation_id=%s AND created_at>=%s ORDER BY created_at ASC",
                    (session["conversation_id"], session["started_at"]),
                )
            else:
                cur.execute(
                    "SELECT * FROM messages WHERE conversation_id=%s AND created_at>=%s AND created_at<=%s ORDER BY created_at ASC",
                    (session["conversation_id"], session["started_at"], session["ended_at"]),
                )
            return cur.fetchall()

    def save_handover_summary(self, handover_session_id: int, theme_summary: str, user_state_summary: str, project_state_summary: str, tag_change_summary: str, human_strategy_summary: str, resume_suggestion: str, summary_json: dict) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO handover_summaries (
                        handover_session_id, theme_summary, user_state_summary, project_state_summary,
                        tag_change_summary, human_strategy_summary, resume_suggestion, summary_json, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON CONFLICT (handover_session_id)
                    DO UPDATE SET theme_summary=EXCLUDED.theme_summary,
                                  user_state_summary=EXCLUDED.user_state_summary,
                                  project_state_summary=EXCLUDED.project_state_summary,
                                  tag_change_summary=EXCLUDED.tag_change_summary,
                                  human_strategy_summary=EXCLUDED.human_strategy_summary,
                                  resume_suggestion=EXCLUDED.resume_suggestion,
                                  summary_json=EXCLUDED.summary_json
                    """,
                    (handover_session_id, theme_summary, user_state_summary, project_state_summary, tag_change_summary, human_strategy_summary, resume_suggestion, summary_json),
                )

    def get_latest_handover_summary_by_conversation(self, conversation_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT hs.*, h.conversation_id FROM handover_summaries hs
                JOIN handover_sessions h ON hs.handover_session_id=h.id
                WHERE h.conversation_id=%s ORDER BY hs.created_at DESC LIMIT 1
                """,
                (conversation_id,),
            )
            return cur.fetchone()

    def list_handover_sessions_by_user(self, business_account_id: int, user_id: int, limit: int = 10) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT h.* FROM handover_sessions h
                JOIN conversations c ON h.conversation_id=c.id
                WHERE c.business_account_id=%s AND c.user_id=%s
                ORDER BY h.started_at DESC LIMIT %s
                """,
                (business_account_id, user_id, limit),
            )
            return cur.fetchall()


# =========================
# llm + prompts
# =========================

class OpenAIClientAdapter:
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def generate(self, model: str, prompt: str, temperature: float = 0.7, max_output_tokens: int = 800) -> str:
        response = self.client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        text_parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text_parts.append(getattr(content, "text", ""))
        return "".join(text_parts).strip()


class LLMService:
    def __init__(self, client_adapter: OpenAIClientAdapter, model_name: str) -> None:
        self.client_adapter = client_adapter
        self.model_name = model_name

    def classify_user_message(self, prompt: str) -> dict[str, Any]:
        return self._parse_json_result(
            self._call_text_model(prompt, 0.1, 700),
            {
                "user_intent": "daily_chat",
                "need_type": "conversation",
                "emotion_state": None,
                "boundary_signal": None,
                "resistance_signal": None,
                "product_interest_signal": False,
                "explicit_product_query": False,
                "high_intent_signal": False,
                "current_mainline_should_continue": "daily",
                "recommended_chat_mode": "daily",
                "recommended_goal": "keep_replying",
                "reason": "Fallback understanding result.",
            },
        )

    def generate_reply(self, prompt: str) -> str:
        return (self._call_text_model(prompt, 0.8, 500) or "").strip()

    def summarize_handover(self, prompt: str) -> dict[str, Any]:
        return self._parse_json_result(
            self._call_text_model(prompt, 0.2, 700),
            {
                "theme_summary": "",
                "user_state_summary": "",
                "project_state_summary": "",
                "tag_change_summary": "",
                "human_strategy_summary": "",
                "resume_suggestion": "",
            },
        )

    def _call_text_model(self, prompt: str, temperature: float, max_output_tokens: int) -> str:
        return self.client_adapter.generate(self.model_name, prompt, temperature, max_output_tokens)

    @staticmethod
    def _parse_json_result(raw_text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            text = (raw_text or "").strip()
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]
                text = text.removeprefix("json").strip()
            return json.loads(text)
        except Exception:
            return fallback


def build_understanding_prompt(latest_user_message: str, recent_context: list[dict[str, Any]], persona_core_summary: str, user_state_summary: str) -> str:
    recent_messages = [
        {"sender_type": m.get("sender_type"), "message_type": m.get("message_type"), "content_text": m.get("content_text")}
        for m in recent_context[-10:]
    ]
    return f"""
You are a conversation understanding engine for a Telegram business account AI chat assistant.
Return JSON only.
Persona core summary: {persona_core_summary}
User state summary: {user_state_summary}
Recent messages: {json.dumps(recent_messages, ensure_ascii=False)}
Latest user message: {latest_user_message}
Schema:
{{
  "user_intent": "daily_chat | product_consult | emotional_expression | pause_request | objection | high_intent_action",
  "need_type": "conversation | product_info | process | returns | safety | emotional_support | pause | objection | next_step",
  "emotion_state": "positive | neutral | low | anxious | resistant | null",
  "boundary_signal": "busy | later | sleep | stop | null",
  "resistance_signal": "low | medium | high | null",
  "product_interest_signal": true,
  "explicit_product_query": false,
  "high_intent_signal": false,
  "current_mainline_should_continue": "daily | product | emotional | pause",
  "recommended_chat_mode": "daily | product | emotional | pause | high_intent | maintain",
  "recommended_goal": "keep_replying | answer_product | emotional_support | pause_respect | stabilize_and_prepare_human",
  "reason": "brief reason"
}}
""".strip()


def build_reply_prompt(latest_user_message: str, recent_context: list[dict[str, Any]], persona_summary: str, user_state_summary: str, stage: str, chat_mode: str, understanding: dict[str, Any], reply_plan: dict[str, Any], selected_content: dict[str, Any]) -> str:
    recent_messages = [{"sender_type": m.get("sender_type"), "content_text": m.get("content_text")} for m in recent_context[-10:]]
    return f"""
You are the live chat brain for a Telegram business account.
Write exactly one natural reply message in plain text.
Persona summary: {persona_summary}
User state summary: {user_state_summary}
Stage: {stage}
Chat mode: {chat_mode}
Understanding: {json.dumps(understanding, ensure_ascii=False)}
Reply plan: {json.dumps(reply_plan, ensure_ascii=False)}
Selected content: {json.dumps(selected_content, ensure_ascii=False)}
Recent messages: {json.dumps(recent_messages, ensure_ascii=False)}
Latest user message: {latest_user_message}
Rules:
- If mode is pause, respect the boundary and do not continue product pushing.
- If mode is emotional, acknowledge emotion first.
- If mode is product or high_intent, answer the user's product question directly and stay on the product mainline.
- Avoid sounding robotic, over-perfect, or overly salesy.
""".strip()


def build_handover_summary_prompt(pre_handover_state: dict[str, Any], handover_messages: list[dict[str, Any]]) -> str:
    normalized_messages = [
        {"sender_type": i.get("sender_type"), "message_type": i.get("message_type"), "content_text": i.get("content_text"), "created_at": str(i.get("created_at"))}
        for i in handover_messages
    ]
    return f"""
You are summarizing a human takeover session for an AI chat assistant.
Pre-handover state: {json.dumps(pre_handover_state, ensure_ascii=False)}
Messages during handover: {json.dumps(normalized_messages, ensure_ascii=False)}
Return JSON only:
{{
  "theme_summary": "what was discussed",
  "user_state_summary": "current user emotional/interest state",
  "project_state_summary": "project progress state",
  "tag_change_summary": "important tag-relevant changes",
  "human_strategy_summary": "what strategy the human used",
  "resume_suggestion": "continue_product | stabilize | wait | maintain"
}}
""".strip()


# =========================
# sender / transport
# =========================

class TelegramBotAPIClient:
    def __init__(self, bot_token: str, db: Database, admin_chat_ids: list[int] | None = None) -> None:
        self.bot_token = bot_token
        self.db = db
        self.admin_chat_ids = admin_chat_ids or []
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def _call_api(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        clean_payload = {k: v for k, v in payload.items() if v is not None}
        data = json.dumps(clean_payload).encode("utf-8")
        req = urllib_request.Request(
            url=f"{self.base_url}/{method}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Telegram API HTTP error on {method}: {e.code} {body}") from e
        except URLError as e:
            raise RuntimeError(f"Telegram API URL error on {method}: {e}") from e

        parsed = json.loads(body)
        if not parsed.get("ok"):
            raise RuntimeError(f"Telegram API call failed on {method}: {parsed}")
        return parsed

    def ensure_webhook(self, webhook_url: str, secret_token: str = "") -> dict[str, Any] | None:
        if not webhook_url:
            return None
        payload: dict[str, Any] = {
            "url": webhook_url,
            "allowed_updates": [
                "business_connection",
                "business_message",
                "edited_business_message",
                "deleted_business_messages",
                "message",
                "callback_query",
            ],
            "drop_pending_updates": False,
        }
        if secret_token:
            payload["secret_token"] = secret_token
        return self._call_api("setWebhook", payload)

    def answer_callback_query(self, callback_query_id: str) -> None:
        self._call_api("answerCallbackQuery", {"callback_query_id": callback_query_id})

    def send_admin_text(self, chat_id: int, text: str) -> None:
        self._call_api("sendMessage", {"chat_id": chat_id, "text": text})

    def send_admin_message(self, chat_id: int, text: str, reply_markup=None) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup:
            payload["reply_markup"] = {"inline_keyboard": reply_markup}
        self._call_api("sendMessage", payload)

    def send_text(self, conversation_id: int, text: str) -> None:
        delivery = self._get_delivery_context_for_conversation(conversation_id)
        payload: dict[str, Any] = {
            "chat_id": delivery["chat_id"],
            "text": text,
            "business_connection_id": delivery.get("business_connection_id"),
        }
        self._call_api("sendMessage", payload)

    def _get_delivery_context_for_conversation(self, conversation_id: int) -> dict[str, Any]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT raw_payload_json
                FROM messages
                WHERE conversation_id = %s
                  AND sender_type = 'user'
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (conversation_id,),
            )
            rows = cur.fetchall()

        for row in rows:
            payload = row.get("raw_payload_json") or {}
            business_connection_id = _extract_business_connection_id(payload)
            chat_id = _extract_chat_id(payload)
            if chat_id is not None:
                return {
                    "chat_id": chat_id,
                    "business_connection_id": business_connection_id,
                }

        raise RuntimeError(f"No delivery context found for conversation_id={conversation_id}")


def _extract_business_connection_id(raw_update: dict[str, Any]) -> str | None:
    if raw_update.get("business_connection_id"):
        return str(raw_update["business_connection_id"])

    for key in ("business_message", "edited_business_message"):
        msg = raw_update.get(key) or {}
        if msg.get("business_connection_id"):
            return str(msg["business_connection_id"])

    conn = raw_update.get("business_connection") or {}
    if conn.get("id"):
        return str(conn["id"])
    return None


def _extract_chat_id(raw_update: dict[str, Any]) -> int | None:
    if raw_update.get("chat_id") is not None:
        try:
            return int(raw_update["chat_id"])
        except Exception:
            return None

    for key in ("business_message", "edited_business_message", "message"):
        msg = raw_update.get(key) or {}
        chat = msg.get("chat") or {}
        if chat.get("id") is not None:
            try:
                return int(chat["id"])
            except Exception:
                return None
    return None


def _extract_message_id(raw_update: dict[str, Any]) -> int | None:
    for key in ("business_message", "edited_business_message", "message"):
        msg = raw_update.get(key) or {}
        if msg.get("message_id") is not None:
            try:
                return int(msg["message_id"])
            except Exception:
                return None
    return None


class TelegramBusinessSenderAdapter:
    def __init__(self, tg_client) -> None:
        self.tg_client = tg_client

    def send_text(self, conversation_id: int, text: str) -> None:
        self.tg_client.send_text(conversation_id=conversation_id, text=text)


class SenderService:
    def __init__(self, sender_adapter: TelegramBusinessSenderAdapter) -> None:
        self.sender_adapter = sender_adapter

    def send_text_reply(self, conversation_id: int, text: str, delay_seconds: int = 0) -> None:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        self.sender_adapter.send_text(conversation_id, text)


class AdminNotifier:
    def __init__(self, tg_client, admin_chat_ids: list[int] | None = None) -> None:
        self.tg_client = tg_client
        self.admin_chat_ids = admin_chat_ids or []

    def notify_high_intent(self, business_account_id: int, user_id: int, conversation_id: int | None, level: str, reason: str) -> None:
        if not self.admin_chat_ids:
            return
        text = (
            "🔥 High Intent Alert\n\n"
            f"Business Account ID: {business_account_id}\n"
            f"User ID: {user_id}\n"
            f"Conversation ID: {conversation_id or '-'}\n"
            f"Level: {level}\n"
            f"Reason: {reason}"
        )
        for chat_id in self.admin_chat_ids:
            self.tg_client.send_admin_text(chat_id=chat_id, text=text)


# =========================
# gateway
# =========================

class TelegramUpdateMapper:
    def map_update(self, raw_update: dict) -> InboundMessage | None:
        message_obj = None
        if raw_update.get("business_message"):
            message_obj = raw_update["business_message"]
        elif raw_update.get("edited_business_message"):
            message_obj = raw_update["edited_business_message"]
        else:
            return None

        tg_business_account_id = _extract_business_connection_id(raw_update) or "default_business_connection"
        chat = message_obj.get("chat") or {}
        from_user = message_obj.get("from") or {}

        text = message_obj.get("text") or message_obj.get("caption")
        message_type = self._detect_message_type(message_obj)
        sent_at = self._extract_sent_at(message_obj.get("date"))

        normalized_payload = dict(raw_update)
        normalized_payload["business_connection_id"] = _extract_business_connection_id(raw_update)
        normalized_payload["chat_id"] = chat.get("id")
        normalized_payload["message_id"] = message_obj.get("message_id")

        return InboundMessage(
            business_account_id=1,
            tg_business_account_id=tg_business_account_id,
            user_id=int(from_user.get("id") or chat.get("id") or 0),
            tg_user_id=str(from_user.get("id") or chat.get("id") or "0"),
            conversation_id=None,
            sender_type="user",
            message_type=message_type,
            text=text,
            media_url=None,
            raw_payload=normalized_payload,
            sent_at=sent_at,
        )

    @staticmethod
    def _detect_message_type(message_obj: dict[str, Any]) -> str:
        if message_obj.get("voice"):
            return "voice"
        if message_obj.get("photo"):
            return "image"
        if message_obj.get("video"):
            return "video"
        if message_obj.get("document"):
            return "file"
        if message_obj.get("animation"):
            return "gif"
        return "text"

    @staticmethod
    def _extract_sent_at(ts: Any) -> datetime:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        return utc_now()


class TelegramBusinessGateway:
    def __init__(self, settings: Settings, message_handler) -> None:
        self.settings = settings
        self.message_handler = message_handler
        self.mapper = TelegramUpdateMapper()

    def handle_raw_update(self, raw_update: dict) -> bool:
        inbound = self.mapper.map_update(raw_update)
        if inbound is None:
            return False
        self.message_handler(inbound)
        return True
# =========================
# core / business / content
# =========================

class PersonaCore:
    def __init__(self) -> None:
        self.gender = "female"
        self.age_feel = 33
        self.role = "financial_wealth_planning_advisor"
        self.professional_vibe = "light_business"
        self.relationship_style = "warm_and_low_pressure"
        self.tone_base = "natural"
        self.tone_pressure = "low_pressure"
        self.marketing_model = "light_marketing"
        self.self_sharing_enabled = True
        self.daily_life_talk_enabled = True

    def to_summary(self) -> str:
        return (
            f"Gender: {self.gender}; Age feel: {self.age_feel}; Role: {self.role}; "
            f"Professional vibe: {self.professional_vibe}; Relationship style: {self.relationship_style}; "
            f"Tone: {self.tone_base}; Pressure: {self.tone_pressure}; Marketing: {self.marketing_model}."
        )


class PersonaProfileBuilder:
    def __init__(self, material_repo: MaterialRepository) -> None:
        self.material_repo = material_repo

    def build(self, business_account_id: int) -> dict:
        display_name = self.material_repo.get_business_account_display_name(business_account_id)
        materials = self.material_repo.get_persona_materials(business_account_id)
        bio_short_parts: list[str] = []
        self_share_topics: list[str] = []
        daily_life_style: list[str] = []
        for item in materials:
            material_type = item.get("material_type")
            content_text = item.get("content_text") or ""
            scene_tags = item.get("scene_tags_json") or []
            if material_type in ("intro", "resume") and content_text:
                bio_short_parts.append(content_text[:200])
            if material_type == "daily":
                if content_text:
                    self_share_topics.append(content_text[:120])
                if scene_tags:
                    daily_life_style.extend(scene_tags)
        return {
            "public_name": display_name,
            "nickname": display_name,
            "bio_short": " | ".join(bio_short_parts[:2]).strip() or f"{display_name} is a warm, low-pressure financial planning advisor.",
            "bio_long": " ".join(bio_short_parts[:4]).strip(),
            "self_share_topics": self_share_topics[:8],
            "daily_life_style": daily_life_style[:12],
        }

    @staticmethod
    def to_summary(profile: dict) -> str:
        return (
            f"Public name: {profile.get('public_name')}; Nickname: {profile.get('nickname')}; "
            f"Bio short: {profile.get('bio_short')}; Self share topics: {profile.get('self_share_topics')}; "
            f"Daily life style: {profile.get('daily_life_style')}."
        )


class UserUnderstandingEngine:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def analyze(self, latest_user_message: str, recent_context: list[dict], persona_core_summary: str, user_state_summary: str) -> UnderstandingResult:
        fallback = self._fallback_rule_based(latest_user_message)
        try:
            result = self.llm_service.classify_user_message(
                build_understanding_prompt(latest_user_message, recent_context, persona_core_summary, user_state_summary)
            )
        except Exception:
            result = fallback
        merged = dict(fallback)
        for k, v in result.items():
            if v is not None:
                merged[k] = v
        return UnderstandingResult(**merged)

    def _fallback_rule_based(self, text: str) -> dict:
        t = (text or "").strip().lower()
        busy_words = ["busy", "later", "sleep", "sleeping", "work", "working", "tomorrow"]
        emotional_words = ["tired", "upset", "sad", "stress", "stressed", "annoyed", "frustrated", "exhausted"]
        product_words = ["project", "product", "returns", "profit", "safety", "safe", "process", "how", "start", "join"]
        high_intent_words = ["how do i start", "how to start", "next step", "what do i do", "can i start", "start now"]
        boundary_signal = None
        if any(w in t for w in busy_words):
            boundary_signal = "sleep" if ("sleep" in t or "sleeping" in t) else ("later" if ("later" in t or "tomorrow" in t) else "busy")
        emotion_state = "low" if any(w in t for w in emotional_words) else None
        explicit_product_query = any(w in t for w in product_words)
        high_intent_signal = any(w in t for w in high_intent_words)
        product_interest_signal = explicit_product_query or high_intent_signal
        if boundary_signal:
            return {
                "user_intent": "pause_request", "need_type": "pause", "emotion_state": emotion_state,
                "boundary_signal": boundary_signal, "resistance_signal": None,
                "product_interest_signal": product_interest_signal, "explicit_product_query": explicit_product_query,
                "high_intent_signal": high_intent_signal, "current_mainline_should_continue": "pause",
                "recommended_chat_mode": "pause", "recommended_goal": "pause_respect", "reason": "Fallback rule-based understanding."
            }
        if high_intent_signal:
            return {
                "user_intent": "high_intent_action", "need_type": "next_step", "emotion_state": emotion_state,
                "boundary_signal": None, "resistance_signal": None,
                "product_interest_signal": True, "explicit_product_query": explicit_product_query,
                "high_intent_signal": True, "current_mainline_should_continue": "product",
                "recommended_chat_mode": "high_intent", "recommended_goal": "stabilize_and_prepare_human", "reason": "Fallback rule-based understanding."
            }
        if emotion_state:
            return {
                "user_intent": "emotional_expression", "need_type": "emotional_support", "emotion_state": emotion_state,
                "boundary_signal": None, "resistance_signal": None,
                "product_interest_signal": product_interest_signal, "explicit_product_query": explicit_product_query,
                "high_intent_signal": False, "current_mainline_should_continue": "emotional",
                "recommended_chat_mode": "emotional", "recommended_goal": "emotional_support", "reason": "Fallback rule-based understanding."
            }
        if explicit_product_query:
            return {
                "user_intent": "product_consult", "need_type": "product_info", "emotion_state": None,
                "boundary_signal": None, "resistance_signal": None,
                "product_interest_signal": True, "explicit_product_query": True,
                "high_intent_signal": False, "current_mainline_should_continue": "product",
                "recommended_chat_mode": "product", "recommended_goal": "answer_product", "reason": "Fallback rule-based understanding."
            }
        return {
            "user_intent": "daily_chat", "need_type": "conversation", "emotion_state": None,
            "boundary_signal": None, "resistance_signal": None,
            "product_interest_signal": False, "explicit_product_query": False,
            "high_intent_signal": False, "current_mainline_should_continue": "daily",
            "recommended_chat_mode": "daily", "recommended_goal": "keep_replying", "reason": "Fallback rule-based understanding."
        }


class ConversationStageEngine:
    def decide(self, understanding: UnderstandingResult, current_stage: str | None) -> StageDecision:
        target = current_stage or "new"
        reason = "keep current stage"
        if understanding.high_intent_signal:
            target = "conversion"
            reason = "high intent detected"
        elif understanding.explicit_product_query or understanding.product_interest_signal:
            if current_stage not in ("conversion",):
                target = "project"
                reason = "product discussion detected"
        elif current_stage in (None, "", "new"):
            target = "relationship"
            reason = "default relationship stage after first contact"
        return StageDecision(stage=target, changed=(current_stage != target), reason=reason)


class ChatModeRouter:
    def decide(self, understanding: UnderstandingResult, current_mode: str | None) -> ModeDecision:
        if understanding.boundary_signal:
            target, reason = "pause", f"boundary detected: {understanding.boundary_signal}"
        elif understanding.high_intent_signal:
            target, reason = "high_intent", "high intent signal"
        elif understanding.emotion_state in ("low", "anxious") and not understanding.explicit_product_query:
            target, reason = "emotional", "emotion needs support first"
        elif understanding.explicit_product_query or understanding.product_interest_signal:
            target, reason = "product", "product mainline"
        elif current_mode == "product" and understanding.current_mainline_should_continue == "product":
            target, reason = "product", "keep product mainline stable"
        else:
            target, reason = understanding.recommended_chat_mode or current_mode or "daily", understanding.reason or "use recommended mode"
        return ModeDecision(chat_mode=target, changed=(current_mode != target), reason=reason)


class ReplyPlanner:
    def plan(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> ReplyPlan:
        _ = stage_decision
        mode = mode_decision.chat_mode
        plan = ReplyPlan(goal=understanding.recommended_goal, reason=f"mode={mode}")
        if mode == "pause":
            plan.should_continue_product = False
            plan.should_leave_space = True
            plan.goal = "pause_respect"
            return plan
        if mode == "emotional":
            plan.should_continue_product = False
            plan.goal = "emotional_support"
            return plan
        if mode == "product":
            plan.should_continue_product = True
            plan.should_send_material = bool(understanding.explicit_product_query and understanding.need_type in ("product_info", "returns", "process", "safety"))
            plan.goal = "answer_product"
            return plan
        if mode == "high_intent":
            plan.should_continue_product = True
            plan.should_prepare_human_escalation = True
            plan.goal = "stabilize_and_prepare_human"
            return plan
        if mode == "maintain":
            plan.should_leave_space = True
            plan.goal = "maintain"
            return plan
        plan.should_self_share = True if "product_interest" not in user_state.tags else False
        return plan


class ReplyStyleEngine:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def generate(self, latest_user_message: str, recent_context: list[dict], persona_summary: str, user_state_summary: str, stage: str, chat_mode: str, understanding: dict, reply_plan: dict, selected_content: dict) -> str:
        try:
            text = self.llm_service.generate_reply(build_reply_prompt(
                latest_user_message, recent_context, persona_summary, user_state_summary,
                stage, chat_mode, understanding, reply_plan, selected_content,
            )).strip()
            if text:
                return text
        except Exception:
            pass
        if chat_mode == "pause":
            return "No worries, take your time. We can talk later when you're free."
        if chat_mode == "emotional":
            return "I get that. Take it one step at a time, and don't push yourself too hard right now."
        if chat_mode == "high_intent":
            return "You're asking the key part now. I’ll help you keep this clear and simple from here."
        if chat_mode == "product":
            return "Sure, I can explain this part clearly for you."
        return "I’m here. Tell me a little more and I’ll follow your pace."


class ReplySelfCheckEngine:
    def check_and_fix(self, draft_reply: str, mode: str, understanding: UnderstandingResult) -> str:
        text = (draft_reply or "").strip() or "I’m here."
        for phrase in ["dear customer", "customer service", "please be informed", "thank you for contacting us"]:
            text = text.replace(phrase, "")
        if mode == "pause" and len(text) > 160:
            text = text[:160].rstrip() + "."
        if mode == "emotional" and not understanding.explicit_product_query:
            for risky in ["returns", "profit", "start now", "join now"]:
                text = text.replace(risky, "")
        if len(text) > 600:
            text = text[:600].rstrip() + "..."
        return text.strip()


class ReplyDelayEngine:
    def decide_delay_seconds(self, understanding: UnderstandingResult, mode: str, reply_text: str) -> int:
        _ = understanding
        _ = reply_text
        return {"pause": 1, "high_intent": 2, "product": 3, "emotional": 3, "maintain": 2}.get(mode, 2)


class AISwitchEngine:
    def __init__(self, settings_repo: SettingsRepository, user_control_repo: UserControlRepository) -> None:
        self.settings_repo = settings_repo
        self.user_control_repo = user_control_repo

    def decide(self, business_account_id: int, conversation_id: int, ops_category: str, manual_takeover_status: str) -> tuple[bool, str]:
        if not self.settings_repo.get_global_ai_enabled(business_account_id):
            return False, "global ai disabled"
        if not self.settings_repo.get_ops_category_ai_enabled(business_account_id, ops_category):
            return False, f"ops category disabled: {ops_category}"
        control = self.user_control_repo.get_user_ai_control(conversation_id)
        if control:
            if control.get("manual_takeover_forced"):
                return False, "manual takeover forced"
            if is_silence_active(control.get("silence_until")):
                return False, "silence active"
            override = control.get("ai_enabled_override")
            if override is False:
                return False, "conversation ai override disabled"
            if override is True:
                return True, "conversation ai override enabled"
        if manual_takeover_status == "active":
            return False, "conversation in manual handover"
        return True, "ai enabled"


class ProjectClassifier:
    def __init__(self, project_repo: ProjectRepository) -> None:
        self.project_repo = project_repo

    def classify(self, business_account_id: int, understanding: UnderstandingResult, context: ConversationContext, user_state: UserStateSnapshot, latest_user_message: str) -> ProjectDecision:
        if user_state.project_id:
            if understanding.explicit_product_query or understanding.product_interest_signal or context.current_chat_mode in ("product", "high_intent"):
                return ProjectDecision(project_id=user_state.project_id, confidence=0.9, source="existing", changed=False, reason="keep existing project stable")
        projects = self.project_repo.list_active_projects(business_account_id)
        if not projects:
            return ProjectDecision(project_id=None, confidence=0.0, source="none", changed=False, reason="no active projects")
        text = ((latest_user_message or "") + " " + (understanding.need_type or "")).lower()
        scored = []
        for p in projects:
            score = 0.0
            name = (p.get("name") or "").lower()
            desc = (p.get("description") or "").lower()
            if name and name in text:
                score += 0.75
            for token in name.split():
                if token and len(token) > 1 and token in text:
                    score += 0.15
            for token in desc.split():
                if token and len(token) > 4 and token in text:
                    score += 0.04
            if understanding.product_interest_signal:
                score += 0.08
            if understanding.high_intent_signal:
                score += 0.05
            scored.append({"project_id": p["id"], "project_name": p.get("name"), "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        if best["score"] >= 0.72:
            return ProjectDecision(project_id=best["project_id"], candidate_projects=scored[:3], confidence=best["score"], source="ai", changed=True, reason=f"matched project: {best['project_name']}")
        if best["score"] >= 0.28:
            return ProjectDecision(project_id=None, candidate_projects=scored[:3], confidence=best["score"], source="ai", changed=False, reason="candidate projects available, no hard switch")
        return ProjectDecision(project_id=None, confidence=0.0, source="none", changed=False, reason="no confident project match")


class IntentEngine:
    def detect(self, understanding: UnderstandingResult, context: ConversationContext, user_state: UserStateSnapshot, project_decision: ProjectDecision) -> IntentDecision:
        score = 0.0
        if understanding.product_interest_signal:
            score += 0.20
        if understanding.explicit_product_query:
            score += 0.20
        if understanding.current_mainline_should_continue == "product":
            score += 0.10
        if project_decision.project_id is not None:
            score += 0.05
        if context.current_chat_mode in ("product", "high_intent"):
            score += 0.10
        if user_state.ops_category == "followup_user":
            score += 0.05
        elif user_state.ops_category == "high_intent_user":
            score += 0.10
        if understanding.high_intent_signal:
            score += 0.30
        if understanding.need_type == "next_step":
            score += 0.15
        if understanding.boundary_signal:
            score -= 0.20
        if understanding.resistance_signal == "medium":
            score -= 0.15
        elif understanding.resistance_signal == "high":
            score -= 0.30
        score = max(0.0, min(1.0, score))
        level = "high" if score >= 0.75 else ("mid" if score >= 0.40 else "low")
        return IntentDecision(score=round(score, 4), level=level, reason=f"intent score: {score:.2f}")


class ProjectSegmentManager:
    def __init__(self, project_repo: ProjectRepository) -> None:
        self.project_repo = project_repo

    def decide(self, project_id: int | None, understanding: UnderstandingResult, intent_decision: IntentDecision) -> dict:
        if not project_id:
            return {"project_segment_id": None, "segment_name": None, "reason": "no project"}
        segments = self.project_repo.get_project_segments(project_id)
        segment_name = "initial_understanding"
        if intent_decision.level == "high":
            segment_name = "high_intent_progress"
        elif understanding.need_type == "returns":
            segment_name = "asked_returns"
        elif understanding.need_type in ("process", "next_step"):
            segment_name = "asked_process"
        elif understanding.need_type == "objection" or understanding.resistance_signal in ("medium", "high"):
            segment_name = "in_objection"
        matched = next((s for s in segments if s.get("name") == segment_name), None)
        return {"project_segment_id": matched["id"] if matched else None, "segment_name": segment_name, "reason": f"segment by need_type={understanding.need_type}"}


class TaggingEngine:
    def decide(self, understanding: UnderstandingResult, intent_decision: IntentDecision, escalation_decision: dict, existing_tags: list[str] | None = None) -> TagDecision:
        existing_tags = existing_tags or []
        add_tags: list[str] = []
        remove_tags: list[str] = []
        if understanding.boundary_signal:
            add_tags.append("recently_busy")
        if understanding.product_interest_signal:
            add_tags.append("product_interest")
        if understanding.resistance_signal in ("medium", "high"):
            add_tags.append("cautious")
        if intent_decision.level in ("mid", "high"):
            add_tags.append("followup_worthy")
        if intent_decision.level == "high":
            add_tags.append("high_intent")
        if escalation_decision.get("should_notify_human"):
            add_tags.append("need_human")
        if not understanding.boundary_signal and "recently_busy" in existing_tags:
            remove_tags.append("recently_busy")
        return TagDecision(add_tags=list(dict.fromkeys(add_tags)), remove_tags=list(dict.fromkeys(remove_tags)), reason="minimal core tag rules applied")


class HumanEscalationEngine:
    def decide(self, understanding: UnderstandingResult, intent_decision: IntentDecision) -> dict:
        if understanding.high_intent_signal and intent_decision.level == "high":
            return {"should_notify_human": True, "notify_level": "urgent_takeover", "should_queue_admin": True, "reason": "high intent next-step signal"}
        if intent_decision.level == "high":
            return {"should_notify_human": True, "notify_level": "suggest_takeover", "should_queue_admin": True, "reason": "high intent detected"}
        if intent_decision.level == "mid":
            return {"should_notify_human": True, "notify_level": "watch", "should_queue_admin": True, "reason": "mid intent needs watching"}
        return {"should_notify_human": False, "notify_level": None, "should_queue_admin": False, "reason": "no escalation needed"}


class OpsCategoryManager:
    PROTECTED = {"archived_user", "invalid_user", "blacklist_user"}
    def decide(self, current_ops_category: str, intent_decision: IntentDecision, understanding: UnderstandingResult) -> dict:
        if current_ops_category in self.PROTECTED:
            return {"ops_category": current_ops_category, "changed": False, "reason": "protected ops category"}
        target = current_ops_category or "new_user"
        if intent_decision.level == "high":
            target = "high_intent_user"
        elif intent_decision.level == "mid" or understanding.product_interest_signal:
            target = "followup_user"
        elif current_ops_category in (None, "", "new_user"):
            target = "new_user"
        return {"ops_category": target, "changed": target != current_ops_category, "reason": "minimal ops category decision"}


class ScriptSelector:
    CATEGORY_MAP = {
        "product": ["intro", "selling_point", "deep_explanation", "objection_handling"],
        "high_intent": ["deep_explanation", "push_forward", "objection_handling"],
    }
    def __init__(self, script_repo: ScriptRepository) -> None:
        self.script_repo = script_repo
    def select(self, project_id: int | None, mode: str) -> list[dict]:
        if not project_id or mode not in self.CATEGORY_MAP:
            return []
        return self.script_repo.get_project_scripts(project_id, self.CATEGORY_MAP[mode])[:3]


class MaterialSelector:
    def __init__(self, material_repo: MaterialRepository) -> None:
        self.material_repo = material_repo
    def select(self, project_id: int | None, mode: str, should_send_material: bool) -> list[dict]:
        if not project_id or not should_send_material or mode not in ("product", "high_intent"):
            return []
        return self.material_repo.get_project_materials(project_id)[:1]


class PersonaMaterialSelector:
    def select(self, persona_materials: list[dict], mode: str, limit: int = 2) -> list[dict]:
        if mode not in ("daily", "maintain", "emotional"):
            return []
        picked: list[dict] = []
        for item in persona_materials:
            if item.get("material_type") == "daily":
                picked.append(item)
            elif item.get("material_type") in ("intro", "resume") and mode == "daily":
                picked.append(item)
            if len(picked) >= limit:
                break
        return picked


class ContentSelector:
    def __init__(self, material_repo: MaterialRepository, script_selector: ScriptSelector, material_selector: MaterialSelector, persona_material_selector: PersonaMaterialSelector) -> None:
        self.material_repo = material_repo
        self.script_selector = script_selector
        self.material_selector = material_selector
        self.persona_material_selector = persona_material_selector
    def select(self, business_account_id: int, project_id: int | None, mode: str, reply_plan: ReplyPlan) -> dict:
        persona_materials = self.material_repo.get_persona_materials(business_account_id)
        return {
            "persona_materials": self.persona_material_selector.select(persona_materials, mode),
            "project_scripts": self.script_selector.select(project_id, mode),
            "project_materials": self.material_selector.select(project_id, mode, reply_plan.should_send_material),
        }


# =========================
# handover
# =========================

class HandoverManager:
    def __init__(self, handover_repo: HandoverRepository, user_control_repo: UserControlRepository, conversation_repo: ConversationRepository, admin_queue_repo: AdminQueueRepository) -> None:
        self.handover_repo = handover_repo
        self.user_control_repo = user_control_repo
        self.conversation_repo = conversation_repo
        self.admin_queue_repo = admin_queue_repo

    def start_handover(self, business_account_id: int, user_id: int, conversation_id: int, started_by: str, handover_reason: str) -> dict:
        active = self.handover_repo.get_active_handover_session(conversation_id)
        if active:
            return {"ok": True, "already_active": True, "session": active, "reason": "handover already active"}
        session = self.handover_repo.create_handover_session(conversation_id, started_by, handover_reason)
        self.user_control_repo.set_manual_takeover(conversation_id, True, handover_reason, started_by)
        self.conversation_repo.set_manual_takeover_status(conversation_id, "active")
        for item in self.admin_queue_repo.list_open_items(business_account_id, limit=100):
            if int(item["user_id"]) == int(user_id):
                self.admin_queue_repo.mark_processing(int(item["id"]))
        return {"ok": True, "already_active": False, "session": session, "reason": "handover started"}

    def end_handover(self, conversation_id: int, session_id: int, ended_by: str) -> dict:
        active = self.handover_repo.get_active_handover_session(conversation_id)
        if not active:
            return {"ok": False, "reason": "no active handover session"}
        if int(active["id"]) != int(session_id):
            return {"ok": False, "reason": "session mismatch"}
        self.handover_repo.end_handover_session(session_id, ended_by)
        self.user_control_repo.set_manual_takeover(conversation_id, False, "handover ended, waiting resume", ended_by)
        self.conversation_repo.set_manual_takeover_status(conversation_id, "pending_resume")
        return {"ok": True, "reason": "handover ended and conversation is pending resume"}


class HandoverSummaryBuilder:
    def __init__(self, handover_repo: HandoverRepository, conversation_repo: ConversationRepository, llm_service: LLMService) -> None:
        self.handover_repo = handover_repo
        self.conversation_repo = conversation_repo
        self.llm_service = llm_service

    def build_summary(self, handover_session_id: int, conversation_id: int) -> dict:
        context = self.conversation_repo.get_context(conversation_id)
        handover_messages = self.handover_repo.get_handover_messages(handover_session_id)
        pre_handover_state = {
            "conversation_id": context.conversation_id,
            "business_account_id": context.business_account_id,
            "user_id": context.user_id,
            "current_stage": context.current_stage,
            "current_chat_mode": context.current_chat_mode,
            "current_mainline": context.current_mainline,
            "manual_takeover_status": context.manual_takeover_status,
            "recent_summary": context.recent_summary,
        }
        try:
            summary = self.llm_service.summarize_handover(build_handover_summary_prompt(pre_handover_state, handover_messages))
        except Exception:
            summary = {
                "theme_summary": "manual handover finished",
                "user_state_summary": "state needs manual review",
                "project_state_summary": "project status updated during handover",
                "tag_change_summary": "",
                "human_strategy_summary": "",
                "resume_suggestion": "maintain",
            }
        self.handover_repo.save_handover_summary(
            handover_session_id,
            summary.get("theme_summary", ""),
            summary.get("user_state_summary", ""),
            summary.get("project_state_summary", ""),
            summary.get("tag_change_summary", ""),
            summary.get("human_strategy_summary", ""),
            summary.get("resume_suggestion", ""),
            summary,
        )
        return summary


class ResumeChatManager:
    def __init__(self, handover_repo: HandoverRepository, conversation_repo: ConversationRepository, user_control_repo: UserControlRepository) -> None:
        self.handover_repo = handover_repo
        self.conversation_repo = conversation_repo
        self.user_control_repo = user_control_repo

    def get_resume_decision(self, conversation_id: int) -> ResumeDecision:
        context = self.conversation_repo.get_context(conversation_id)
        latest_summary = self.handover_repo.get_latest_handover_summary_by_conversation(conversation_id)
        if context.manual_takeover_status != "pending_resume":
            return ResumeDecision(False, context.current_chat_mode or "maintain", "wait", "", "conversation is not pending resume")
        resume_suggestion = latest_summary.get("resume_suggestion") if latest_summary else ""
        mode = context.current_chat_mode or "maintain"
        goal = "maintain"
        opening_hint = "continue naturally from the latest human conversation context"
        if resume_suggestion == "continue_product":
            mode, goal, opening_hint = "product", "answer_product", "continue product mainline naturally without repeating old questions"
        elif resume_suggestion == "stabilize":
            mode, goal, opening_hint = "maintain", "stabilize_and_prepare_human", "stabilize the conversation first"
        elif resume_suggestion == "wait":
            mode, goal, opening_hint = "maintain", "wait", "do not rush, wait for a natural continuation"
        return ResumeDecision(True, mode, goal, opening_hint, "resume decision built from latest handover summary")

    def resume_ai(self, conversation_id: int, operator: str) -> dict:
        context = self.conversation_repo.get_context(conversation_id)
        if context.manual_takeover_status != "pending_resume":
            return {"ok": False, "reason": "conversation is not pending resume"}
        self.conversation_repo.set_manual_takeover_status(conversation_id, "inactive")
        self.user_control_repo.clear_ai_override(conversation_id, operator)
        return {"ok": True, "reason": "ai resumed"}


# =========================
# orchestrator
# =========================

class Orchestrator:
    def __init__(self, conversation_repo: ConversationRepository, user_repo: UserRepository, settings_repo: SettingsRepository,
                 user_control_repo: UserControlRepository, business_account_repo: BusinessAccountRepository,
                 bootstrap_repo: BootstrapRepository, receipt_repo: ReceiptRepository, admin_queue_repo: AdminQueueRepository,
                 material_repo: MaterialRepository, project_repo: ProjectRepository, script_repo: ScriptRepository,
                 persona_core: PersonaCore, persona_profile_builder: PersonaProfileBuilder,
                 understanding_engine: UserUnderstandingEngine, stage_engine: ConversationStageEngine,
                 mode_router: ChatModeRouter, reply_planner: ReplyPlanner, reply_style_engine: ReplyStyleEngine,
                 reply_self_check_engine: ReplySelfCheckEngine, reply_delay_engine: ReplyDelayEngine,
                 ai_switch_engine: AISwitchEngine, project_classifier: ProjectClassifier,
                 project_segment_manager: ProjectSegmentManager, tagging_engine: TaggingEngine,
                 intent_engine: IntentEngine, human_escalation_engine: HumanEscalationEngine,
                 ops_category_manager: OpsCategoryManager, content_selector: ContentSelector,
                 sender_service: SenderService, handover_repo: HandoverRepository | None = None,
                 admin_notifier: AdminNotifier | None = None) -> None:
        self.conversation_repo = conversation_repo
        self.user_repo = user_repo
        self.settings_repo = settings_repo
        self.user_control_repo = user_control_repo
        self.business_account_repo = business_account_repo
        self.bootstrap_repo = bootstrap_repo
        self.receipt_repo = receipt_repo
        self.admin_queue_repo = admin_queue_repo
        self.material_repo = material_repo
        self.project_repo = project_repo
        self.script_repo = script_repo
        self.persona_core = persona_core
        self.persona_profile_builder = persona_profile_builder
        self.understanding_engine = understanding_engine
        self.stage_engine = stage_engine
        self.mode_router = mode_router
        self.reply_planner = reply_planner
        self.reply_style_engine = reply_style_engine
        self.reply_self_check_engine = reply_self_check_engine
        self.reply_delay_engine = reply_delay_engine
        self.ai_switch_engine = ai_switch_engine
        self.project_classifier = project_classifier
        self.project_segment_manager = project_segment_manager
        self.tagging_engine = tagging_engine
        self.intent_engine = intent_engine
        self.human_escalation_engine = human_escalation_engine
        self.ops_category_manager = ops_category_manager
        self.content_selector = content_selector
        self.sender_service = sender_service
        self.handover_repo = handover_repo
        self.admin_notifier = admin_notifier

    def handle_inbound_message(self, inbound_message: InboundMessage) -> None:
        business_account = self.business_account_repo.create_if_not_exists(
            inbound_message.tg_business_account_id,
            f"BusinessAccount-{inbound_message.tg_business_account_id}",
            None,
        )
        business_account_id = int(business_account["id"])
        self.bootstrap_repo.ensure_default_ai_control_settings(business_account_id)
        self.bootstrap_repo.ensure_default_persona_core(business_account_id)
        self.bootstrap_repo.ensure_default_tags(business_account_id)
        user_id = self.user_repo.get_or_create_user(inbound_message.tg_user_id, f"User-{inbound_message.tg_user_id}")
        conversation_id = self.conversation_repo.get_or_create_conversation(business_account_id, user_id)
        self.conversation_repo.save_message(conversation_id, "user", inbound_message.message_type, inbound_message.text, inbound_message.raw_payload, inbound_message.media_url)
        context = self.conversation_repo.get_context(conversation_id)
        user_state = self.user_repo.get_user_state_snapshot(business_account_id, user_id)
        ai_allowed, ai_reason = self.ai_switch_engine.decide(business_account_id, conversation_id, user_state.ops_category, context.manual_takeover_status)
        if not ai_allowed:
            logger.info("AI reply skipped | conversation_id=%s | reason=%s", conversation_id, ai_reason)
            return
        latest_user_text = inbound_message.text or ""
        persona_profile = self.persona_profile_builder.build(business_account_id)
        persona_summary = self.persona_core.to_summary() + " " + self.persona_profile_builder.to_summary(persona_profile)
        recent_context = list(context.recent_messages)
        latest_handover_summary = None
        if self.handover_repo and context.manual_takeover_status == "pending_resume":
            latest_handover_summary = self.handover_repo.get_latest_handover_summary_by_conversation(conversation_id)
            if latest_handover_summary:
                recent_context.append({
                    "sender_type": "system",
                    "message_type": "text",
                    "content_text": f"[ResumeHint] theme={latest_handover_summary.get('theme_summary')}; user_state={latest_handover_summary.get('user_state_summary')}; resume={latest_handover_summary.get('resume_suggestion')}"
                })
        understanding = self.understanding_engine.analyze(latest_user_text, recent_context, self.persona_core.to_summary(), self._user_state_summary(user_state))
        stage_decision = self.stage_engine.decide(understanding, context.current_stage)
        mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
        project_decision = self.project_classifier.classify(business_account_id, understanding, context, user_state, latest_user_text)
        intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
        segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
        escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
        tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
        ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)
        reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
        selected_content = self.content_selector.select(business_account_id, project_decision.project_id, mode_decision.chat_mode, reply_plan)
        understanding_payload = understanding.__dict__.copy()
        if latest_handover_summary:
            understanding_payload["resume_hint"] = latest_handover_summary.get("resume_suggestion")
        draft_reply = self.reply_style_engine.generate(
            latest_user_text, recent_context, persona_summary, self._user_state_summary(user_state),
            stage_decision.stage, mode_decision.chat_mode, understanding_payload,
            reply_plan.__dict__, selected_content,
        )
        final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        final_reply = FinalReply(text=final_text, delay_seconds=delay_seconds)
        if reply_plan.should_reply and final_reply.text.strip():
            self.sender_service.send_text_reply(conversation_id, final_reply.text, final_reply.delay_seconds)
            self.conversation_repo.save_message(conversation_id, "ai", "text", final_reply.text, {"selected_content": selected_content}, None)
            self.conversation_repo.set_last_ai_reply_at(conversation_id)
        self.conversation_repo.update_conversation_state(conversation_id, stage_decision.stage, mode_decision.chat_mode, understanding.current_mainline_should_continue)
        if project_decision.project_id is not None:
            self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
        self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_decision["ops_category"], ops_decision["reason"], "system")
        if escalation_decision.get("should_queue_admin"):
            queue_type = "urgent_handover" if escalation_decision.get("notify_level") == "urgent_takeover" else "high_intent"
            priority_score = 95.0 if queue_type == "urgent_handover" else (80.0 if escalation_decision.get("notify_level") == "suggest_takeover" else 60.0)
            self.admin_queue_repo.upsert_queue_item(business_account_id, user_id, queue_type, priority_score, escalation_decision["reason"])
            self.receipt_repo.create_high_intent_receipt(
                business_account_id, user_id, "High intent detected",
                {
                    "notify_level": escalation_decision.get("notify_level"),
                    "reason": escalation_decision.get("reason"),
                    "project_id": project_decision.project_id,
                    "segment_name": segment_decision.get("segment_name"),
                    "intent_level": intent_decision.level,
                    "intent_score": intent_decision.score,
                },
            )
            if self.admin_notifier:
                self.admin_notifier.notify_high_intent(business_account_id, user_id, conversation_id, escalation_decision.get("notify_level") or "watch", escalation_decision.get("reason") or "")

    @staticmethod
    def _user_state_summary(user_state: UserStateSnapshot) -> str:
        return (
            f"ops_category={user_state.ops_category}; project_id={user_state.project_id}; "
            f"project_segment_id={user_state.project_segment_id}; tags={user_state.tags}; "
            f"relationship_score={user_state.relationship_score}; trust_score={user_state.trust_score}; "
            f"comfort_score={user_state.comfort_score}; current_heat={user_state.current_heat}; "
            f"marketing_tolerance={user_state.marketing_tolerance}"
        )


# =========================
# admin services + tg admin
# =========================

class CustomerActions:
    def __init__(self, user_control_repo: UserControlRepository, user_repo: UserRepository, handover_manager: HandoverManager, handover_summary_builder: HandoverSummaryBuilder, resume_chat_manager: ResumeChatManager, handover_repo: HandoverRepository, conversation_repo: ConversationRepository) -> None:
        self.user_control_repo = user_control_repo
        self.user_repo = user_repo
        self.handover_manager = handover_manager
        self.handover_summary_builder = handover_summary_builder
        self.resume_chat_manager = resume_chat_manager
        self.handover_repo = handover_repo
        self.conversation_repo = conversation_repo

    def enable_ai(self, conversation_id: int, operator: str) -> dict:
        self.user_control_repo.set_ai_override(conversation_id, True, "manual enable ai", operator)
        return {"ok": True, "reason": "ai enabled"}

    def disable_ai(self, conversation_id: int, operator: str) -> dict:
        self.user_control_repo.set_ai_override(conversation_id, False, "manual disable ai", operator)
        return {"ok": True, "reason": "ai disabled"}

    def start_handover(self, business_account_id: int, user_id: int, conversation_id: int, operator: str, reason: str = "manual handover") -> dict:
        return self.handover_manager.start_handover(business_account_id, user_id, conversation_id, operator, reason)

    def end_handover(self, conversation_id: int, session_id: int, operator: str) -> dict:
        result = self.handover_manager.end_handover(conversation_id, session_id, operator)
        if not result.get("ok"):
            return result
        summary = self.handover_summary_builder.build_summary(session_id, conversation_id)
        return {"ok": True, "reason": "handover ended and summary generated", "summary": summary}

    def resume_ai(self, conversation_id: int, operator: str) -> dict:
        return self.resume_chat_manager.resume_ai(conversation_id, operator)

    def set_project(self, business_account_id: int, user_id: int, project_id: int | None, operator: str, reason_text: str = "manual project update") -> dict:
        self.user_repo.set_project_manual(business_account_id, user_id, project_id, reason_text, operator)
        return {"ok": True, "reason": "project updated"}

    def add_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str = "manual tag add") -> dict:
        self.user_repo.add_manual_tag(business_account_id, user_id, tag_name, operator, reason_text)
        return {"ok": True, "reason": "tag added"}

    def remove_tag(self, business_account_id: int, user_id: int, tag_name: str) -> dict:
        self.user_repo.remove_manual_tag(business_account_id, user_id, tag_name)
        return {"ok": True, "reason": "tag removed"}

    def set_ops_category(self, business_account_id: int, user_id: int, ops_category: str, operator: str, reason_text: str = "manual ops category update") -> dict:
        self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_category, reason_text, operator)
        return {"ok": True, "reason": "ops category updated"}


class AdminAPIService:
    def __init__(self, user_repo: UserRepository, receipt_repo: ReceiptRepository, handover_repo: HandoverRepository, conversation_repo: ConversationRepository, user_control_repo: UserControlRepository, admin_queue_repo: AdminQueueRepository, customer_actions: CustomerActions, resume_chat_manager: ResumeChatManager, project_repo: ProjectRepository | None = None) -> None:
        self.user_repo = user_repo
        self.receipt_repo = receipt_repo
        self.handover_repo = handover_repo
        self.conversation_repo = conversation_repo
        self.user_control_repo = user_control_repo
        self.admin_queue_repo = admin_queue_repo
        self.customer_actions = customer_actions
        self.resume_chat_manager = resume_chat_manager
        self.project_repo = project_repo

    def get_customer_detail(self, business_account_id: int, user_id: int) -> dict:
        profile = self.user_repo.get_user_profile(business_account_id, user_id)
        user_row = profile.get("user") or {}
        if not user_row:
            return {"ok": False, "reason": "user not found"}
        conversation_id = self._find_conversation_id(business_account_id, user_id)
        recent_messages = []
        current_ai_control = None
        latest_handover_summary = None
        context_payload = None
        if conversation_id:
            context = self.conversation_repo.get_context(conversation_id)
            recent_messages = context.recent_messages
            current_ai_control = self.user_control_repo.get_user_ai_control(conversation_id)
            latest_handover_summary = self.handover_repo.get_latest_handover_summary_by_conversation(conversation_id)
            context_payload = context.__dict__
        receipts = self.receipt_repo.list_recent_receipts(business_account_id, user_id, 10)
        handover_history = self.handover_repo.list_handover_sessions_by_user(business_account_id, user_id, 10)
        queue_items = self.admin_queue_repo.list_items_by_user(business_account_id, user_id, 20)
        resume_decision = None
        if conversation_id and context_payload and context_payload.get("manual_takeover_status") == "pending_resume":
            resume_decision = self.resume_chat_manager.get_resume_decision(conversation_id).__dict__
        return {
            "ok": True,
            "profile": profile,
            "conversation": {"conversation_id": conversation_id, "context": context_payload, "recent_messages": recent_messages},
            "ai_control": current_ai_control,
            "receipts": receipts,
            "handover_history": handover_history,
            "latest_handover_summary": latest_handover_summary,
            "queue_items": queue_items,
            "resume_decision": resume_decision,
        }

    def get_customer_detail_by_conversation(self, conversation_id: int) -> dict:
        db = self.conversation_repo.db
        with db.cursor() as cur:
            cur.execute("SELECT business_account_id, user_id FROM conversations WHERE id=%s LIMIT 1", (conversation_id,))
            row = cur.fetchone()
        if not row:
            return {"ok": False, "reason": "conversation not found"}
        return self.get_customer_detail(int(row["business_account_id"]), int(row["user_id"]))

    def list_projects_for_business_account(self, business_account_id: int) -> list[dict]:
        return [] if not self.project_repo else self.project_repo.list_active_projects(business_account_id)

    def list_all_tag_names(self, business_account_id: int) -> list[str]:
        db = self.conversation_repo.db
        with db.cursor() as cur:
            cur.execute("SELECT name FROM tags WHERE business_account_id=%s AND is_active=TRUE ORDER BY name ASC", (business_account_id,))
            rows = cur.fetchall()
        return [r["name"] for r in rows]

    def list_active_tag_names_for_user(self, business_account_id: int, user_id: int) -> list[str]:
        detail = self.get_customer_detail(business_account_id, user_id)
        return [] if not detail.get("ok") else ((detail.get("profile", {}) or {}).get("tags", []) or [])

    def list_pending_resume_items(self, business_account_id: int) -> list[dict]:
        db = self.conversation_repo.db
        with db.cursor() as cur:
            cur.execute(
                """
                SELECT c.id AS conversation_id, c.user_id, c.current_chat_mode,
                       ups.project_id, p.name AS project_name
                FROM conversations c
                LEFT JOIN user_project_state ups ON ups.business_account_id=c.business_account_id AND ups.user_id=c.user_id
                LEFT JOIN projects p ON ups.project_id=p.id
                WHERE c.business_account_id=%s AND c.manual_takeover_status='pending_resume'
                ORDER BY c.updated_at DESC LIMIT 50
                """,
                (business_account_id,),
            )
            return cur.fetchall()

    def start_manual_handover(self, business_account_id: int, user_id: int, conversation_id: int, operator: str, reason: str = "manual handover") -> dict:
        return self.customer_actions.start_handover(business_account_id, user_id, conversation_id, operator, reason)

    def end_manual_handover(self, conversation_id: int, session_id: int, operator: str) -> dict:
        return self.customer_actions.end_handover(conversation_id, session_id, operator)

    def enable_ai_for_conversation(self, conversation_id: int, operator: str) -> dict:
        return self.customer_actions.enable_ai(conversation_id, operator)

    def disable_ai_for_conversation(self, conversation_id: int, operator: str) -> dict:
        return self.customer_actions.disable_ai(conversation_id, operator)

    def resume_ai(self, conversation_id: int, operator: str) -> dict:
        return self.customer_actions.resume_ai(conversation_id, operator)

    def set_project(self, business_account_id: int, user_id: int, project_id: int | None, operator: str, reason_text: str = "manual project update") -> dict:
        return self.customer_actions.set_project(business_account_id, user_id, project_id, operator, reason_text)

    def add_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str = "manual tag add") -> dict:
        return self.customer_actions.add_tag(business_account_id, user_id, tag_name, operator, reason_text)

    def remove_tag(self, business_account_id: int, user_id: int, tag_name: str) -> dict:
        return self.customer_actions.remove_tag(business_account_id, user_id, tag_name)

    def set_ops_category(self, business_account_id: int, user_id: int, ops_category: str, operator: str, reason_text: str = "manual ops category update") -> dict:
        return self.customer_actions.set_ops_category(business_account_id, user_id, ops_category, operator, reason_text)

    def _find_conversation_id(self, business_account_id: int, user_id: int) -> int | None:
        db = self.conversation_repo.db
        with db.cursor() as cur:
            cur.execute("SELECT id FROM conversations WHERE business_account_id=%s AND user_id=%s LIMIT 1", (business_account_id, user_id))
            row = cur.fetchone()
            return None if not row else int(row["id"])


class DashboardService:
    def __init__(self, settings_repo: SettingsRepository, admin_queue_repo: AdminQueueRepository, receipt_repo: ReceiptRepository, handover_repo: HandoverRepository, conversation_repo: ConversationRepository) -> None:
        self.settings_repo = settings_repo
        self.admin_queue_repo = admin_queue_repo
        self.receipt_repo = receipt_repo
        self.handover_repo = handover_repo
        self.conversation_repo = conversation_repo

    def get_summary(self, business_account_id: int) -> dict:
        open_queue = self.admin_queue_repo.list_open_items(business_account_id, limit=200)
        high_intent_queue = [q for q in open_queue if q["queue_type"] == "high_intent"]
        urgent_queue = [q for q in open_queue if q["queue_type"] == "urgent_handover"]
        return {
            "ai_global_enabled": self.settings_repo.get_global_ai_enabled(business_account_id),
            "ops_counts": self._count_ops_categories(business_account_id),
            "pending_receipt_count": self.receipt_repo.count_pending_receipts(business_account_id),
            "open_admin_queue_count": len(open_queue),
            "high_intent_queue_count": len(high_intent_queue),
            "urgent_queue_count": len(urgent_queue),
            "active_handover_count": self._count_active_handover(business_account_id),
            "pending_resume_count": self._count_pending_resume(business_account_id),
        }

    def list_open_queue(self, business_account_id: int, queue_type: str | None = None, limit: int = 50) -> list[dict]:
        return self.admin_queue_repo.list_open_items(business_account_id, queue_type, limit)

    def _count_pending_resume(self, business_account_id: int) -> int:
        db = self.conversation_repo.db
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM conversations WHERE business_account_id=%s AND manual_takeover_status='pending_resume'", (business_account_id,))
            row = cur.fetchone()
            return int(row["cnt"] or 0)

    def _count_active_handover(self, business_account_id: int) -> int:
        db = self.conversation_repo.db
        with db.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt FROM handover_sessions h
                JOIN conversations c ON h.conversation_id=c.id
                WHERE c.business_account_id=%s AND h.status='active'
                """,
                (business_account_id,),
            )
            row = cur.fetchone()
            return int(row["cnt"] or 0)

    def _count_ops_categories(self, business_account_id: int) -> dict:
        db = self.conversation_repo.db
        with db.cursor() as cur:
            cur.execute("SELECT ops_category, COUNT(*) AS cnt FROM user_ops_status WHERE business_account_id=%s GROUP BY ops_category", (business_account_id,))
            rows = cur.fetchall()
        result = {"new_user": 0, "followup_user": 0, "high_intent_user": 0, "archived_user": 0, "invalid_user": 0, "blacklist_user": 0}
        for row in rows:
            result[row["ops_category"]] = int(row["cnt"])
        return result


# TG admin format/menu/callbacks/handlers simplified
class TGAdminFormatter:
    @staticmethod
    def format_dashboard(summary: dict) -> str:
        ops = summary.get("ops_counts", {}) or {}
        return (
            "📈 Dashboard Summary\n\n"
            f"AI Global: {'ON' if summary.get('ai_global_enabled') else 'OFF'}\n"
            f"Pending Receipts: {summary.get('pending_receipt_count', 0)}\n"
            f"Open Queue: {summary.get('open_admin_queue_count', 0)}\n"
            f"High Intent Queue: {summary.get('high_intent_queue_count', 0)}\n"
            f"Urgent Queue: {summary.get('urgent_queue_count', 0)}\n"
            f"Active Handover: {summary.get('active_handover_count', 0)}\n"
            f"Pending Resume: {summary.get('pending_resume_count', 0)}\n\n"
            "Ops Counts:\n"
            f"- New: {ops.get('new_user', 0)}\n"
            f"- Follow-up: {ops.get('followup_user', 0)}\n"
            f"- High Intent: {ops.get('high_intent_user', 0)}"
        )


class TGAdminMenuBuilder:
    @staticmethod
    def main_menu() -> dict:
        return {
            "text": "📊 Admin Console\nChoose an action:",
            "reply_markup": [
                [{"text": "📈 Dashboard", "callback_data": "adm:dashboard"}, {"text": "🔥 High Intent", "callback_data": "adm:queue:high_intent"}],
                [{"text": "⚠️ Urgent", "callback_data": "adm:queue:urgent_handover"}, {"text": "⏳ Pending Resume", "callback_data": "adm:queue:pending_resume"}],
                [{"text": "📂 Open Queue", "callback_data": "adm:queue:all"}],
            ],
        }


class TGAdminCallbackRouter:
    def __init__(self, admin_api_service: AdminAPIService, dashboard_service: DashboardService, tg_sender) -> None:
        self.admin_api_service = admin_api_service
        self.dashboard_service = dashboard_service
        self.tg_sender = tg_sender

    def handle(self, admin_chat_id: int, callback_data: str, operator: str = "admin") -> None:
        parts = (callback_data or "").split(":")
        if not parts or parts[0] != "adm":
            self.tg_sender.send_admin_text(chat_id=admin_chat_id, text="❌ Unknown admin callback.")
            return
        action = parts[1] if len(parts) > 1 else "main"
        if action == "main":
            menu = TGAdminMenuBuilder.main_menu()
            self.tg_sender.send_admin_message(admin_chat_id, menu["text"], menu["reply_markup"])
            return
        if action == "dashboard":
            summary = self.dashboard_service.get_summary(1)
            self.tg_sender.send_admin_text(admin_chat_id, TGAdminFormatter.format_dashboard(summary))
            return
        self.tg_sender.send_admin_text(admin_chat_id, f"⚠️ Unsupported callback: {callback_data}")


class TGAdminHandlers:
    def __init__(self, tg_admin_callback_router: TGAdminCallbackRouter, tg_sender) -> None:
        self.tg_admin_callback_router = tg_admin_callback_router
        self.tg_sender = tg_sender

    def handle_admin_command(self, admin_chat_id: int, command_text: str, operator: str = "admin") -> None:
        text = (command_text or "").strip().lower()
        if text in ("/admin", "/startadmin", "admin"):
            menu = TGAdminMenuBuilder.main_menu()
            self.tg_sender.send_admin_message(admin_chat_id, menu["text"], menu["reply_markup"])
            return
        self.tg_sender.send_admin_text(admin_chat_id, "⚠️ Unknown admin command. Use /admin")

    def handle_admin_callback(self, admin_chat_id: int, callback_data: str, operator: str = "admin") -> None:
        self.tg_admin_callback_router.handle(admin_chat_id, callback_data, operator)


# =========================
# web admin routes
# =========================

def build_admin_blueprint(admin_api_service: AdminAPIService, dashboard_service: DashboardService) -> Blueprint:
    bp = Blueprint("admin", __name__, url_prefix="/admin")

    @bp.get("/health")
    def health():
        return jsonify({"ok": True, "service": "admin"})

    @bp.get("/dashboard/summary")
    def dashboard_summary():
        business_account_id = request.args.get("business_account_id", type=int)
        if not business_account_id:
            return jsonify({"ok": False, "error": "business_account_id is required"}), 400
        return jsonify({"ok": True, "data": dashboard_service.get_summary(business_account_id)})

    @bp.get("/queue/open")
    def queue_open():
        business_account_id = request.args.get("business_account_id", type=int)
        queue_type = request.args.get("queue_type", default=None, type=str)
        limit = request.args.get("limit", default=50, type=int)
        if not business_account_id:
            return jsonify({"ok": False, "error": "business_account_id is required"}), 400
        return jsonify({"ok": True, "data": dashboard_service.list_open_queue(business_account_id, queue_type, limit)})

    @bp.get("/customers/<int:business_account_id>/<int:user_id>")
    def customer_detail(business_account_id: int, user_id: int):
        data = admin_api_service.get_customer_detail(business_account_id, user_id)
        return jsonify(data), (200 if data.get("ok") else 404)

    @bp.post("/conversations/<int:conversation_id>/ai/enable")
    def enable_ai(conversation_id: int):
        payload = request.get_json(silent=True) or {}
        data = admin_api_service.enable_ai_for_conversation(conversation_id, payload.get("operator", "admin"))
        return jsonify(data), (200 if data.get("ok") else 400)

    @bp.post("/conversations/<int:conversation_id>/ai/disable")
    def disable_ai(conversation_id: int):
        payload = request.get_json(silent=True) or {}
        data = admin_api_service.disable_ai_for_conversation(conversation_id, payload.get("operator", "admin"))
        return jsonify(data), (200 if data.get("ok") else 400)

    @bp.post("/conversations/<int:conversation_id>/resume-ai")
    def resume_ai(conversation_id: int):
        payload = request.get_json(silent=True) or {}
        data = admin_api_service.resume_ai(conversation_id, payload.get("operator", "admin"))
        return jsonify(data), (200 if data.get("ok") else 400)

    return bp


def _is_admin_chat_allowed(settings: Settings, chat_id: int) -> bool:
    if not settings.admin_chat_ids:
        return True
    return chat_id in settings.admin_chat_ids


def create_web_app(settings: Settings, app_components: dict[str, Any]) -> Flask:
    flask_app = Flask(__name__)
    admin_api_service = app_components["admin_api_service"]
    dashboard_service = app_components["dashboard_service"]
    gateway = app_components["gateway"]
    tg_admin_handlers = app_components["tg_admin_handlers"]
    tg_client = app_components["tg_client"]
    flask_app.register_blueprint(build_admin_blueprint(admin_api_service, dashboard_service))

    @flask_app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "tg-business-ai-chat"})

    @flask_app.post("/webhook/telegram")
    def telegram_webhook():
        raw_update = request.get_json(silent=True) or {}

        if settings.telegram_webhook_secret_token:
            header_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if not hmac.compare_digest(header_token, settings.telegram_webhook_secret_token):
                return jsonify({"ok": False, "error": "invalid webhook secret"}), 403

        try:
            callback_query = raw_update.get("callback_query") or {}
            if callback_query:
                message = callback_query.get("message") or {}
                chat = message.get("chat") or {}
                admin_chat_id = int(chat.get("id")) if chat.get("id") is not None else None
                if admin_chat_id is not None and _is_admin_chat_allowed(settings, admin_chat_id):
                    tg_admin_handlers.handle_admin_callback(
                        admin_chat_id=admin_chat_id,
                        callback_data=callback_query.get("data") or "",
                        operator=str((callback_query.get("from") or {}).get("id") or "admin"),
                    )
                    if callback_query.get("id"):
                        tg_client.answer_callback_query(str(callback_query["id"]))
                    return jsonify({"ok": True, "dispatched": "admin_callback"})

            message = raw_update.get("message") or {}
            if message and not raw_update.get("business_message") and not raw_update.get("edited_business_message"):
                chat = message.get("chat") or {}
                if chat.get("type") == "private" and chat.get("id") is not None:
                    admin_chat_id = int(chat["id"])
                    if _is_admin_chat_allowed(settings, admin_chat_id):
                        text = message.get("text") or ""
                        operator = str((message.get("from") or {}).get("id") or "admin")
                        if text.startswith("/admin") or text.startswith("/startadmin"):
                            tg_admin_handlers.handle_admin_command(admin_chat_id, text, operator=operator)
                            return jsonify({"ok": True, "dispatched": "admin_command"})
                        tg_admin_handlers.handle_admin_text(admin_chat_id, text, operator=operator)
                        return jsonify({"ok": True, "dispatched": "admin_text"})

            dispatched = gateway.handle_raw_update(raw_update)
            return jsonify({"ok": True, "dispatched": "business_message" if dispatched else "ignored"})
        except Exception as e:
            logger.exception("telegram_webhook failed")
            return jsonify({"ok": False, "error": str(e)}), 500

    return flask_app


# =========================
# app factory
# =========================

def build_app_components(settings: Settings) -> dict[str, Any]:
    db = Database(settings.database_url)
    db.connect()

    tg_client = TelegramBotAPIClient(
        bot_token=settings.tg_bot_token,
        db=db,
        admin_chat_ids=settings.admin_chat_ids,
    )

    business_account_repo = BusinessAccountRepository(db)
    bootstrap_repo = BootstrapRepository(db)
    conversation_repo = ConversationRepository(db)
    user_repo = UserRepository(db)
    settings_repo = SettingsRepository(db)
    user_control_repo = UserControlRepository(db)
    material_repo = MaterialRepository(db)
    project_repo = ProjectRepository(db)
    script_repo = ScriptRepository(db)
    receipt_repo = ReceiptRepository(db)
    admin_queue_repo = AdminQueueRepository(db)
    handover_repo = HandoverRepository(db)

    openai_adapter = OpenAIClientAdapter(settings.openai_api_key)
    llm_service = LLMService(openai_adapter, settings.llm_model_name)
    sender_service = SenderService(TelegramBusinessSenderAdapter(tg_client))
    admin_notifier = AdminNotifier(tg_client, admin_chat_ids=settings.admin_chat_ids)

    persona_core = PersonaCore()
    persona_profile_builder = PersonaProfileBuilder(material_repo)
    understanding_engine = UserUnderstandingEngine(llm_service)
    stage_engine = ConversationStageEngine()
    mode_router = ChatModeRouter()
    reply_planner = ReplyPlanner()
    reply_style_engine = ReplyStyleEngine(llm_service)
    reply_self_check_engine = ReplySelfCheckEngine()
    reply_delay_engine = ReplyDelayEngine()

    ai_switch_engine = AISwitchEngine(settings_repo, user_control_repo)
    project_classifier = ProjectClassifier(project_repo)
    project_segment_manager = ProjectSegmentManager(project_repo)
    tagging_engine = TaggingEngine()
    intent_engine = IntentEngine()
    human_escalation_engine = HumanEscalationEngine()
    ops_category_manager = OpsCategoryManager()

    content_selector = ContentSelector(material_repo, ScriptSelector(script_repo), MaterialSelector(material_repo), PersonaMaterialSelector())

    handover_manager = HandoverManager(handover_repo, user_control_repo, conversation_repo, admin_queue_repo)
    handover_summary_builder = HandoverSummaryBuilder(handover_repo, conversation_repo, llm_service)
    resume_chat_manager = ResumeChatManager(handover_repo, conversation_repo, user_control_repo)

    customer_actions = CustomerActions(user_control_repo, user_repo, handover_manager, handover_summary_builder, resume_chat_manager, handover_repo, conversation_repo)
    admin_api_service = AdminAPIService(user_repo, receipt_repo, handover_repo, conversation_repo, user_control_repo, admin_queue_repo, customer_actions, resume_chat_manager, project_repo)
    dashboard_service = DashboardService(settings_repo, admin_queue_repo, receipt_repo, handover_repo, conversation_repo)
    tg_admin_callback_router = TGAdminCallbackRouter(admin_api_service, dashboard_service, tg_client)
    tg_admin_handlers = TGAdminHandlers(tg_admin_callback_router, tg_client)

    orchestrator = Orchestrator(
        conversation_repo, user_repo, settings_repo, user_control_repo, business_account_repo,
        bootstrap_repo, receipt_repo, admin_queue_repo, material_repo, project_repo, script_repo,
        persona_core, persona_profile_builder, understanding_engine, stage_engine,
        mode_router, reply_planner, reply_style_engine, reply_self_check_engine,
        reply_delay_engine, ai_switch_engine, project_classifier, project_segment_manager,
        tagging_engine, intent_engine, human_escalation_engine, ops_category_manager,
        content_selector, sender_service, handover_repo, admin_notifier,
    )

    gateway = TelegramBusinessGateway(settings, orchestrator.handle_inbound_message)
    return {
        "db": db,
        "llm_service": llm_service,
        "sender_service": sender_service,
        "admin_notifier": admin_notifier,
        "gateway": gateway,
        "orchestrator": orchestrator,
        "handover_manager": handover_manager,
        "handover_summary_builder": handover_summary_builder,
        "resume_chat_manager": resume_chat_manager,
        "customer_actions": customer_actions,
        "admin_api_service": admin_api_service,
        "dashboard_service": dashboard_service,
        "tg_admin_callback_router": tg_admin_callback_router,
        "tg_admin_handlers": tg_admin_handlers,
        "tg_client": tg_client,
    }


def main() -> None:
    settings = Settings.load()
    settings.validate()
    setup_logging(settings.log_level)
    logger.info("Starting application.")
    app_components = build_app_components(settings)

    tg_client = app_components["tg_client"]
    if settings.telegram_webhook_url:
        try:
            tg_client.ensure_webhook(
                webhook_url=settings.telegram_webhook_url,
                secret_token=settings.telegram_webhook_secret_token,
            )
            logger.info("Telegram webhook ensured.")
        except Exception:
            logger.exception("Failed to ensure Telegram webhook.")

    flask_app = create_web_app(settings, app_components)
    logger.info("Application initialized successfully | host=%s | port=%s", settings.webhook_host, settings.webhook_port)
    flask_app.run(host=settings.webhook_host, port=settings.webhook_port)


if __name__ == "__main__":
    main()
