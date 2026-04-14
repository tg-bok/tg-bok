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
    telegram_webhook_url: str
    telegram_webhook_secret_token: str
    admin_chat_ids: list[int]

    @classmethod
    def load(cls) -> "Settings":
        admin_chat_ids_raw = os.getenv("ADMIN_CHAT_IDS", "").strip()
        admin_chat_ids: list[int] = []

        if admin_chat_ids_raw:
            for item in admin_chat_ids_raw.split(","):
                item = item.strip()
                if item:
                    try:
                        admin_chat_ids.append(int(item))
                    except ValueError:
                        pass

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
            telegram_webhook_url=os.getenv("TELEGRAM_WEBHOOK_URL", "").strip(),
            telegram_webhook_secret_token=os.getenv("TELEGRAM_WEBHOOK_SECRET_TOKEN", "").strip(),
            admin_chat_ids=admin_chat_ids,
        )

    def validate(self) -> None:
        missing: list[str] = []
        if not self.database_url:
            missing.append("DATABASE_URL")
        if not self.tg_bot_token:
            missing.append("TG_BOT_TOKEN")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.telegram_webhook_url:
            missing.append("TELEGRAM_WEBHOOK_URL")
        if not self.telegram_webhook_secret_token:
            missing.append("TELEGRAM_WEBHOOK_SECRET_TOKEN")
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

    def rollback(self) -> None:
        if self.conn is not None:
            self.conn.rollback()

    def commit(self) -> None:
        if self.conn is not None:
            self.conn.commit()

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


def initialize_database(db: Database) -> None:
    schema_statements = [
        '''
        CREATE TABLE IF NOT EXISTS business_accounts (
            id BIGSERIAL PRIMARY KEY,
            tg_business_account_id TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            username TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            tg_user_id TEXT UNIQUE NOT NULL,
            username TEXT,
            display_name TEXT,
            language_code TEXT,
            first_seen_at TIMESTAMPTZ,
            last_seen_at TIMESTAMPTZ,
            is_blocked BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS conversations (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            current_stage TEXT,
            current_chat_mode TEXT,
            current_mainline TEXT,
            ai_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            manual_takeover_status TEXT NOT NULL DEFAULT 'inactive',
            last_message_at TIMESTAMPTZ,
            last_ai_reply_at TIMESTAMPTZ,
            last_human_reply_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (business_account_id, user_id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS messages (
            id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            sender_type TEXT NOT NULL,
            message_type TEXT NOT NULL,
            content_text TEXT,
            media_url TEXT,
            raw_payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS ai_control_settings (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL UNIQUE REFERENCES business_accounts(id) ON DELETE CASCADE,
            global_ai_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            new_user_ai_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            followup_user_ai_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            high_intent_ai_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            archived_user_ai_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            invalid_user_ai_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            blacklist_ai_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS business_account_persona_core (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL UNIQUE REFERENCES business_accounts(id) ON DELETE CASCADE,
            gender TEXT,
            age_feel INTEGER,
            role TEXT,
            professional_vibe TEXT,
            relationship_style TEXT,
            tone_base TEXT,
            tone_pressure TEXT,
            marketing_model TEXT,
            self_sharing_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            daily_life_talk_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            project_before_tag BOOLEAN NOT NULL DEFAULT TRUE,
            manual_override_priority BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS tags (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            tag_group TEXT,
            tag_type TEXT,
            description TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (business_account_id, name)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS projects (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            description TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS project_segments (
            id BIGSERIAL PRIMARY KEY,
            project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            description TEXT,
            sort_order INTEGER NOT NULL DEFAULT 0,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (project_id, name)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS project_scripts (
            id BIGSERIAL PRIMARY KEY,
            project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            category TEXT,
            content_text TEXT,
            priority INTEGER NOT NULL DEFAULT 100,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS project_materials (
            id BIGSERIAL PRIMARY KEY,
            project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            material_type TEXT,
            content_text TEXT,
            media_url TEXT,
            scene_tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            priority INTEGER NOT NULL DEFAULT 100,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS persona_materials (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            material_type TEXT,
            content_text TEXT,
            media_url TEXT,
            scene_tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            priority INTEGER NOT NULL DEFAULT 100,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS user_ai_controls (
            id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
            ai_enabled_override BOOLEAN,
            manual_takeover_forced BOOLEAN NOT NULL DEFAULT FALSE,
            silence_until TIMESTAMPTZ,
            reason_text TEXT,
            updated_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS user_ops_status (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            ops_category TEXT NOT NULL,
            reason_text TEXT,
            source TEXT,
            is_locked BOOLEAN NOT NULL DEFAULT FALSE,
            updated_by TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (business_account_id, user_id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS user_project_state (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            project_id BIGINT REFERENCES projects(id) ON DELETE SET NULL,
            candidate_projects_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            source TEXT,
            reason_text TEXT,
            confidence DOUBLE PRECISION,
            status TEXT,
            is_locked BOOLEAN NOT NULL DEFAULT FALSE,
            updated_by TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (business_account_id, user_id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS user_project_segment_state (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            project_segment_id BIGINT REFERENCES project_segments(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (business_account_id, user_id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS user_tags (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            tag_id BIGINT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            source TEXT,
            reason_text TEXT,
            confidence DOUBLE PRECISION,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            is_locked BOOLEAN NOT NULL DEFAULT FALSE,
            expires_at TIMESTAMPTZ,
            created_by TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (business_account_id, user_id, tag_id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS receipts (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            receipt_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content_text TEXT,
            content_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS admin_priority_queue (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            queue_type TEXT NOT NULL,
            priority_score DOUBLE PRECISION NOT NULL DEFAULT 0,
            reason_text TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS handover_sessions (
            id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            started_by TEXT,
            handover_reason TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended_by TEXT,
            ended_at TIMESTAMPTZ
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS handover_summaries (
            id BIGSERIAL PRIMARY KEY,
            handover_session_id BIGINT NOT NULL UNIQUE REFERENCES handover_sessions(id) ON DELETE CASCADE,
            theme_summary TEXT,
            user_state_summary TEXT,
            project_state_summary TEXT,
            tag_change_summary TEXT,
            human_strategy_summary TEXT,
            resume_suggestion TEXT,
            summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            summary_text TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        '''
    ]
    with db.transaction():
        with db.cursor() as cur:
            for stmt in schema_statements:
                cur.execute(stmt)


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
        raw_payload_json = json.dumps(raw_payload or {}, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (conversation_id, sender_type, message_type, content_text, media_url, raw_payload_json)
                    VALUES (%s,%s,%s,%s,%s,%s::jsonb)
                    """,
                    (conversation_id, sender_type, message_type, text, media_url, raw_payload_json),
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
            "📈 仪表盘总览\n\n"
            f"全局AI: {'开启' if summary.get('ai_global_enabled') else '关闭'}\n"
            f"待处理回执: {summary.get('pending_receipt_count', 0)}\n"
            f"开放队列: {summary.get('open_admin_queue_count', 0)}\n"
            f"高意向队列: {summary.get('high_intent_queue_count', 0)}\n"
            f"紧急队列: {summary.get('urgent_queue_count', 0)}\n"
            f"接管中会话: {summary.get('active_handover_count', 0)}\n"
            f"待恢复AI: {summary.get('pending_resume_count', 0)}\n\n"
            "运营分类统计:\n"
            f"- 新用户: {ops.get('new_user', 0)}\n"
            f"- 跟进用户: {ops.get('followup_user', 0)}\n"
            f"- 高意向用户: {ops.get('high_intent_user', 0)}"
        )


class TGAdminMenuBuilder:
    @staticmethod
    def main_menu() -> dict:
        return {
            "text": "📊 管理后台\n请选择操作：",
            "reply_markup": [
                [{"text": "📈 仪表盘", "callback_data": "adm:dashboard"}, {"text": "🔥 高意向客户", "callback_data": "adm:queue:high_intent"}],
                [{"text": "⚠️ 紧急处理", "callback_data": "adm:queue:urgent_handover"}, {"text": "⏳ 待恢复AI", "callback_data": "adm:queue:pending_resume"}],
                [{"text": "📂 打开队列", "callback_data": "adm:queue:all"}],
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
            self.tg_sender.send_admin_text(chat_id=admin_chat_id, text="❌ 未知的管理员回调。")
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
        self.tg_sender.send_admin_text(admin_chat_id, "⚠️ 未知的管理员命令，请使用 /admin")

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
            try:
                app_components["db"].rollback()
            except Exception:
                pass
            logger.exception("telegram_webhook failed")
            return jsonify({"ok": False, "error": str(e)}), 500

    return flask_app


# =========================
# app factory
# =========================

def build_app_components(settings: Settings) -> dict[str, Any]:
    db = Database(settings.database_url)
    db.connect()
    initialize_database(db)

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



# =========================
# V2 upgrade layer
# =========================

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
    boundary_sensitivity: float = 0.25
    response_speed_preference: str = "normal"
    emotional_receptiveness: float = 0.5
    professional_receptiveness: float = 0.5
    recent_busy_score: float = 0.0
    push_fatigue_score: float = 0.0
    openness_score: float = 0.45
    warmth_score: float = 0.45
    current_project_interest_strength: float = 0.0


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
    emotion_strength: float = 0.4
    boundary_strength: float = 0.0
    social_openness: float = 0.45
    project_relevance: float = 0.0
    human_takeover_hint: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class StageDecision:
    stage: str
    changed: bool
    reason: str
    confidence: float = 0.65


@dataclass
class TurnDecision:
    reply_goal: str = "rapport"
    should_push_project: bool = False
    social_distance: str = "balanced"
    reply_length: str = "medium"
    ask_followup_question: bool = False
    exit_strategy: str = "gentle_close"
    self_disclosure_level: str = "light"
    reason: str = ""
    should_reply: bool = True
    marketing_intensity: str = "subtle"
    tone_bias: str = "natural"


@dataclass
class StyleSpec:
    tone: str = "natural"
    formality: str = "balanced"
    warmth: str = "medium"
    length: str = "medium"
    question_rate: str = "low"
    emoji_usage: str = "minimal"
    self_disclosure_ratio: str = "none"
    completion_level: str = "balanced"
    marketing_visibility: str = "subtle"
    naturalness_bias: str = "high"


_Prev_UserRepository_1 = UserRepository
class UserRepository(_Prev_UserRepository_1):
    def get_user_state_snapshot(self, business_account_id: int, user_id: int) -> UserStateSnapshot:
        base = super().get_user_state_snapshot(business_account_id, user_id)
        tags = set(base.tags or [])

        relationship_score = 0.28
        trust_score = 0.24
        comfort_score = 0.34
        current_heat = 0.26
        marketing_tolerance = 0.42
        boundary_sensitivity = 0.28
        response_speed_preference = "normal"
        emotional_receptiveness = 0.52
        professional_receptiveness = 0.52
        recent_busy_score = 0.0
        push_fatigue_score = 0.12
        openness_score = 0.46
        warmth_score = 0.48
        current_project_interest_strength = 0.0

        if "recently_busy" in tags:
            recent_busy_score = 0.88
            boundary_sensitivity = 0.82
            response_speed_preference = "slow"
            marketing_tolerance = min(marketing_tolerance, 0.25)
            push_fatigue_score = max(push_fatigue_score, 0.62)
        if "cautious" in tags:
            trust_score = 0.38
            professional_receptiveness = 0.65
            emotional_receptiveness = 0.45
            marketing_tolerance = min(marketing_tolerance, 0.33)
            boundary_sensitivity = max(boundary_sensitivity, 0.52)
        if "product_interest" in tags:
            current_project_interest_strength = 0.68
            professional_receptiveness = 0.74
            relationship_score = max(relationship_score, 0.44)
            current_heat = max(current_heat, 0.48)
        if "followup_worthy" in tags:
            current_heat = max(current_heat, 0.54)
            openness_score = max(openness_score, 0.58)
        if "high_intent" in tags:
            current_project_interest_strength = 0.9
            trust_score = max(trust_score, 0.7)
            relationship_score = max(relationship_score, 0.66)
            current_heat = max(current_heat, 0.74)
            marketing_tolerance = max(marketing_tolerance, 0.66)
            professional_receptiveness = max(professional_receptiveness, 0.82)
            openness_score = max(openness_score, 0.7)
            warmth_score = max(warmth_score, 0.62)

        if base.ops_category in ("intent_user", "deal_user"):
            relationship_score = max(relationship_score, 0.58)
            trust_score = max(trust_score, 0.6)
            current_heat = max(current_heat, 0.66)
            marketing_tolerance = max(marketing_tolerance, 0.62)

        return UserStateSnapshot(
            ops_category=base.ops_category,
            project_id=base.project_id,
            project_segment_id=base.project_segment_id,
            tags=list(base.tags or []),
            relationship_score=round(relationship_score, 2),
            trust_score=round(trust_score, 2),
            comfort_score=round(comfort_score, 2),
            current_heat=round(current_heat, 2),
            marketing_tolerance=round(marketing_tolerance, 2),
            boundary_sensitivity=round(boundary_sensitivity, 2),
            response_speed_preference=response_speed_preference,
            emotional_receptiveness=round(emotional_receptiveness, 2),
            professional_receptiveness=round(professional_receptiveness, 2),
            recent_busy_score=round(recent_busy_score, 2),
            push_fatigue_score=round(push_fatigue_score, 2),
            openness_score=round(openness_score, 2),
            warmth_score=round(warmth_score, 2),
            current_project_interest_strength=round(current_project_interest_strength, 2),
        )


_Prev_UserUnderstandingEngine_1 = UserUnderstandingEngine
class UserUnderstandingEngine(_Prev_UserUnderstandingEngine_1):
    def analyze(self, latest_user_message: str, recent_context: list[dict], persona_core_summary: str, user_state_summary: str) -> UnderstandingResult:
        fallback = self._fallback_rule_based(latest_user_message)
        try:
            result = self.llm_service.classify_user_message(
                build_understanding_prompt(latest_user_message, recent_context, persona_core_summary, user_state_summary)
            )
        except Exception:
            result = fallback
        merged = dict(fallback)
        for k, v in (result or {}).items():
            if v is not None:
                merged[k] = v
        return UnderstandingResult(**merged)

    def _fallback_rule_based(self, text: str) -> dict:
        t = (text or "").strip().lower()
        busy_words = ["busy", "later", "sleep", "sleeping", "work", "working", "tomorrow", "meeting", "driving"]
        emotional_words = ["tired", "sad", "stress", "stressed", "anxious", "worried", "upset", "rough day"]
        product_words = ["return", "profit", "plan", "project", "rate", "yield", "details", "how does it work", "risk"]
        high_intent_words = ["interested", "want to know", "can i join", "how do i start", "what do i need", "send me details"]
        boundary_signal = None
        boundary_strength = 0.0
        emotion_state = "neutral"
        emotion_strength = 0.35
        if any(w in t for w in busy_words):
            boundary_signal = "busy"
            boundary_strength = 0.9
        elif any(w in t for w in ["not now", "stop", "later please", "leave it", "another time"]):
            boundary_signal = "needs_space"
            boundary_strength = 0.95
        if any(w in t for w in emotional_words):
            emotion_state = "low"
            emotion_strength = 0.76
        elif any(w in t for w in ["good", "nice", "great", "sounds good", "interesting"]):
            emotion_state = "positive"
            emotion_strength = 0.62
        product_interest = any(w in t for w in product_words)
        explicit_product_query = product_interest or "?" in t and any(w in t for w in ["plan", "return", "rate", "risk", "how"])
        high_intent_signal = any(w in t for w in high_intent_words)
        social_openness = 0.38
        if any(w in t for w in ["haha", "lol", "how about you", "what about you", "really", "yeah"]):
            social_openness = 0.62
        if boundary_signal:
            social_openness = min(social_openness, 0.28)
        project_relevance = 0.8 if explicit_product_query else (0.55 if product_interest else 0.15)
        human_takeover_hint = bool(high_intent_signal and any(w in t for w in ["call", "agent", "real person", "someone can explain"]))
        recommended_chat_mode = "pause" if boundary_signal else ("high_intent" if high_intent_signal else ("product" if explicit_product_query else ("emotional" if emotion_state == "low" else "daily")))
        recommended_goal = (
            "respect_boundary" if boundary_signal else
            "progress_without_pressure" if high_intent_signal else
            "answer_and_transition" if explicit_product_query else
            "support_first" if emotion_state == "low" else
            "build_rapport"
        )
        return {
            "user_intent": "product_query" if explicit_product_query else ("interest" if high_intent_signal else "chat"),
            "need_type": "information" if explicit_product_query else ("progress" if high_intent_signal else "rapport"),
            "emotion_state": emotion_state,
            "boundary_signal": boundary_signal,
            "resistance_signal": "soft_resistance" if "not sure" in t or "maybe" in t else None,
            "product_interest_signal": product_interest,
            "explicit_product_query": explicit_product_query,
            "high_intent_signal": high_intent_signal,
            "current_mainline_should_continue": "product" if (explicit_product_query or high_intent_signal) else "daily",
            "recommended_chat_mode": recommended_chat_mode,
            "recommended_goal": recommended_goal,
            "reason": "v2_rule_fallback",
            "emotion_strength": emotion_strength,
            "boundary_strength": boundary_strength,
            "social_openness": social_openness,
            "project_relevance": project_relevance,
            "human_takeover_hint": human_takeover_hint,
            "notes": [],
        }


_Prev_ConversationStageEngine_1 = ConversationStageEngine
class ConversationStageEngine(_Prev_ConversationStageEngine_1):
    def decide(self, understanding: UnderstandingResult, current_stage: str | None) -> StageDecision:
        target = current_stage or "first_contact"
        reason = "retain current stage"
        confidence = 0.66
        if understanding.boundary_signal:
            target = "boundary_protection"
            reason = f"boundary detected: {understanding.boundary_signal}"
            confidence = max(0.82, understanding.boundary_strength)
        elif understanding.high_intent_signal:
            target = "high_intent_progress"
            reason = "high intent detected"
            confidence = 0.88
        elif understanding.explicit_product_query or understanding.project_relevance >= 0.75:
            target = "project_discussion"
            reason = "project discussion detected"
            confidence = 0.79
        elif current_stage in (None, "", "new", "relationship", "first_contact"):
            if understanding.social_openness >= 0.58 or understanding.emotion_state in ("positive", "low"):
                target = "trust_building"
                reason = "rapport or emotional openness detected"
                confidence = 0.71
            else:
                target = "daily_rapport"
                reason = "light rapport maintenance"
                confidence = 0.64
        return StageDecision(stage=target, changed=(current_stage != target), reason=reason, confidence=confidence)


class TurnDecisionEngine:
    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        if understanding.boundary_signal:
            return TurnDecision(
                reply_goal="respect_boundary",
                should_push_project=False,
                social_distance="give_space",
                reply_length="short",
                ask_followup_question=False,
                exit_strategy="leave_space",
                self_disclosure_level="none",
                reason=f"boundary:{understanding.boundary_signal}",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="gentle",
            )
        if understanding.high_intent_signal:
            ask_followup = user_state.boundary_sensitivity < 0.65 and user_state.push_fatigue_score < 0.7
            return TurnDecision(
                reply_goal="progress_without_pressure",
                should_push_project=True,
                social_distance="professional_close",
                reply_length="medium",
                ask_followup_question=ask_followup,
                exit_strategy="soft_next_step",
                self_disclosure_level="none",
                reason="high_intent_signal",
                should_reply=True,
                marketing_intensity="light",
                tone_bias="professional_warm",
            )
        if understanding.explicit_product_query or understanding.project_relevance >= 0.72:
            return TurnDecision(
                reply_goal="answer_and_transition",
                should_push_project=True,
                social_distance="balanced",
                reply_length="medium",
                ask_followup_question=user_state.boundary_sensitivity < 0.55,
                exit_strategy="gentle_close",
                self_disclosure_level="none",
                reason="product_focus",
                should_reply=True,
                marketing_intensity="subtle",
                tone_bias="clear",
            )
        if understanding.emotion_state == "low":
            return TurnDecision(
                reply_goal="support_first",
                should_push_project=False,
                social_distance="warm_supportive",
                reply_length="short",
                ask_followup_question=False,
                exit_strategy="soft_hold",
                self_disclosure_level="none",
                reason="emotion_low",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="soft",
            )
        if stage_decision.stage in ("trust_building", "daily_rapport"):
            return TurnDecision(
                reply_goal="build_rapport",
                should_push_project=False,
                social_distance="friendly_but_measured",
                reply_length="short" if user_state.response_speed_preference == "slow" else "medium",
                ask_followup_question=user_state.boundary_sensitivity < 0.45,
                exit_strategy="light_hook" if user_state.boundary_sensitivity < 0.45 else "gentle_close",
                self_disclosure_level="light" if understanding.social_openness >= 0.55 else "none",
                reason="rapport_stage",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="natural",
            )
        return TurnDecision(
            reply_goal="maintain_presence",
            should_push_project=False,
            social_distance="balanced",
            reply_length="medium",
            ask_followup_question=False,
            exit_strategy="gentle_close",
            self_disclosure_level="none",
            reason="default_v2",
            should_reply=True,
            marketing_intensity="none" if user_state.boundary_sensitivity > 0.55 else "subtle",
            tone_bias="natural",
        )


class HumanizationController:
    def build_style_spec(
        self,
        decision: TurnDecision,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        persona_profile: dict[str, Any] | None = None,
    ) -> StyleSpec:
        _ = persona_profile
        tone = "natural"
        formality = "balanced"
        warmth = "medium"
        question_rate = "low"
        completion_level = "balanced"
        marketing_visibility = decision.marketing_intensity
        self_disclosure_ratio = decision.self_disclosure_level
        if decision.reply_goal == "respect_boundary":
            tone = "gentle"
            formality = "casual"
            warmth = "soft"
            question_rate = "none"
            completion_level = "lean"
        elif decision.reply_goal == "support_first":
            tone = "soft"
            formality = "casual"
            warmth = "high"
            question_rate = "none"
            completion_level = "lean"
        elif decision.reply_goal in ("answer_and_transition", "progress_without_pressure"):
            tone = "clear"
            formality = "balanced"
            warmth = "medium"
            question_rate = "one"
            completion_level = "focused"
        elif stage_decision.stage in ("trust_building", "daily_rapport"):
            tone = "natural"
            formality = "casual"
            warmth = "medium_high"
            question_rate = "low" if not decision.ask_followup_question else "one"
            completion_level = "light"
        if user_state.response_speed_preference == "slow":
            completion_level = "lean"
        if user_state.boundary_sensitivity > 0.7:
            question_rate = "none"
            marketing_visibility = "none"
        return StyleSpec(
            tone=tone,
            formality=formality,
            warmth=warmth,
            length=decision.reply_length,
            question_rate=question_rate,
            emoji_usage="minimal",
            self_disclosure_ratio=self_disclosure_ratio,
            completion_level=completion_level,
            marketing_visibility=marketing_visibility,
            naturalness_bias="high",
        )


_Prev_ReplyStyleEngine_1 = ReplyStyleEngine
class ReplyStyleEngine(_Prev_ReplyStyleEngine_1):
    def generate(self, latest_user_message: str, recent_context: list[dict], persona_summary: str, user_state_summary: str, stage: str, chat_mode: str, understanding: dict, reply_plan: dict, selected_content: dict) -> str:
        text = super().generate(latest_user_message, recent_context, persona_summary, user_state_summary, stage, chat_mode, understanding, reply_plan, selected_content)
        style_spec = understanding.get("style_spec") or {}
        turn_decision = understanding.get("turn_decision") or {}
        return self._apply_style_postprocess(text, style_spec, turn_decision)

    def _apply_style_postprocess(self, text: str, style_spec: dict[str, Any], turn_decision: dict[str, Any]) -> str:
        t = " ".join((text or "").split()).strip()
        if not t:
            return "I’m here."
        length = style_spec.get("length") or "medium"
        completion_level = style_spec.get("completion_level") or "balanced"
        max_len = 340
        if length == "short":
            max_len = 170
        elif length == "medium":
            max_len = 320
        else:
            max_len = 520
        if completion_level == "lean":
            max_len = min(max_len, 180)
        marketing_visibility = style_spec.get("marketing_visibility") or "none"
        if marketing_visibility == "none":
            for frag in ["join now", "start now", "we can get you started", "this is a good time to enter", "profit", "returns"]:
                t = t.replace(frag, "")
                t = t.replace(frag.title(), "")
        question_rate = style_spec.get("question_rate") or "low"
        if question_rate == "none":
            if "?" in t:
                t = t.replace("?", ".")
        elif question_rate == "one":
            first = t.find("?")
            if first != -1:
                later = t.find("?", first + 1)
                if later != -1:
                    t = t[:later].replace("?", ".") + t[later:]
                    t = t.replace("?", ".")
                    if first != -1:
                        lst = list(t)
                        lst[first] = "?"
                        t = "".join(lst)
        if turn_decision.get("exit_strategy") in ("leave_space", "soft_hold") and t.endswith("?"):
            t = t[:-1].rstrip() + "."
        if len(t) > max_len:
            cut = t[:max_len].rstrip()
            if "." in cut[-60:]:
                cut = cut[: cut.rfind(".") + 1]
            else:
                cut += "..."
            t = cut.strip()
        return " ".join(t.split()).strip()


_Prev_ReplySelfCheckEngine_1 = ReplySelfCheckEngine
class ReplySelfCheckEngine(_Prev_ReplySelfCheckEngine_1):
    def check_and_fix(self, draft_reply: str, mode: str, understanding: UnderstandingResult) -> str:
        text = super().check_and_fix(draft_reply, mode, understanding)
        lowered = text.lower()
        banned_fragments = [
            "as an ai", "i am an ai", "customer support", "our platform team", "kindly note",
            "dear user", "please let me know if you have any further questions", "allow me to explain"
        ]
        for frag in banned_fragments:
            if frag in lowered:
                text = text.replace(frag, "")
                text = text.replace(frag.title(), "")
                lowered = text.lower()
        if understanding.boundary_signal:
            for frag in ["can i", "would you like", "shall i", "do you want"]:
                if frag in lowered:
                    text = text.replace(frag, "")
                    lowered = text.lower()
            text = text.replace("??", "?")
            if text.endswith("?"):
                text = text[:-1].rstrip() + "."
        if understanding.emotion_state == "low" and not understanding.explicit_product_query:
            for risky in ["returns", "profit", "join now", "start now", "earn"]:
                text = text.replace(risky, "")
                text = text.replace(risky.title(), "")
        text = " ".join(text.split()).strip()
        if len(text) > 420:
            text = text[:420].rstrip() + "..."
        return text.strip()


_Prev_ReplyDelayEngine_1 = ReplyDelayEngine
class ReplyDelayEngine(_Prev_ReplyDelayEngine_1):
    def decide_delay_seconds(self, understanding: UnderstandingResult, mode: str, reply_text: str) -> int:
        _ = reply_text
        if understanding.boundary_signal:
            return 1
        if mode == "high_intent":
            return 2
        if mode == "product":
            return 3
        if mode == "emotional":
            return 2
        if understanding.social_openness >= 0.58:
            return 2
        return 3


_Prev_Orchestrator_1 = Orchestrator
class Orchestrator(_Prev_Orchestrator_1):
    def __init__(self, *args, turn_decision_engine: TurnDecisionEngine | None = None, humanization_controller: HumanizationController | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.turn_decision_engine = turn_decision_engine or TurnDecisionEngine()
        self.humanization_controller = humanization_controller or HumanizationController()

    def _build_memory_bundle(self, context: ConversationContext, latest_handover_summary: dict[str, Any] | None = None) -> MemoryBundle:
        return MemoryBundle(
            recent_messages=list(context.recent_messages or []),
            recent_summary=context.recent_summary or "",
            long_term_memory=context.long_term_memory or {},
            handover_summary=latest_handover_summary or {},
        )

    def _user_state_summary(self, user_state: UserStateSnapshot) -> str:
        return (
            f"ops_category={user_state.ops_category}; project_id={user_state.project_id}; "
            f"project_segment_id={user_state.project_segment_id}; tags={user_state.tags}; "
            f"relationship_score={user_state.relationship_score}; trust_score={user_state.trust_score}; "
            f"comfort_score={user_state.comfort_score}; current_heat={user_state.current_heat}; "
            f"marketing_tolerance={user_state.marketing_tolerance}; boundary_sensitivity={user_state.boundary_sensitivity}; "
            f"response_speed_preference={user_state.response_speed_preference}; emotional_receptiveness={user_state.emotional_receptiveness}; "
            f"professional_receptiveness={user_state.professional_receptiveness}; recent_busy_score={user_state.recent_busy_score}; "
            f"push_fatigue_score={user_state.push_fatigue_score}; openness_score={user_state.openness_score}; warmth_score={user_state.warmth_score}; "
            f"project_interest_strength={user_state.current_project_interest_strength}"
        )

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

        latest_user_text = (inbound_message.text or "").strip()
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
        memory_bundle = self._build_memory_bundle(context, latest_handover_summary)
        understanding = self.understanding_engine.analyze(latest_user_text, recent_context, self.persona_core.to_summary(), self._user_state_summary(user_state))
        stage_decision = self.stage_engine.decide(understanding, context.current_stage)
        mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
        turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
        style_spec = self.humanization_controller.build_style_spec(turn_decision, understanding, stage_decision, user_state, persona_profile)
        project_decision = self.project_classifier.classify(business_account_id, understanding, context, user_state, latest_user_text)
        intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
        segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
        escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
        escalation_decision["should_queue_admin"] = bool(escalation_decision.get("should_queue_admin") or understanding.human_takeover_hint)
        if understanding.human_takeover_hint and not escalation_decision.get("reason"):
            escalation_decision["reason"] = "user asked for a real person or direct explanation"
            escalation_decision["notify_level"] = "suggest_takeover"
        tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
        ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)
        reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
        reply_plan.should_reply = turn_decision.should_reply
        reply_plan.should_continue_product = turn_decision.should_push_project
        reply_plan.should_leave_space = turn_decision.exit_strategy in ("leave_space", "soft_hold")
        reply_plan.should_self_share = turn_decision.self_disclosure_level != "none"
        selected_content = self.content_selector.select(business_account_id, project_decision.project_id, mode_decision.chat_mode, reply_plan)
        if not turn_decision.should_push_project and not understanding.explicit_product_query:
            selected_content = {k: v for k, v in (selected_content or {}).items() if k in ("persona_materials", "self_share_materials")}
        understanding_payload = understanding.__dict__.copy()
        understanding_payload["turn_decision"] = asdict(turn_decision)
        understanding_payload["style_spec"] = asdict(style_spec)
        understanding_payload["memory_summary"] = {
            "recent_messages_count": len(memory_bundle.recent_messages),
            "has_recent_summary": bool(memory_bundle.recent_summary),
            "has_handover_summary": bool(memory_bundle.handover_summary),
        }
        if latest_handover_summary:
            understanding_payload["resume_hint"] = latest_handover_summary.get("resume_suggestion")
        draft_reply = self.reply_style_engine.generate(
            latest_user_text,
            recent_context,
            persona_summary,
            self._user_state_summary(user_state),
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding_payload,
            reply_plan.__dict__,
            selected_content,
        )
        final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        final_reply = FinalReply(
            text=final_text,
            delay_seconds=delay_seconds,
            metadata={"turn_decision": asdict(turn_decision), "style_spec": asdict(style_spec)},
        )
        if reply_plan.should_reply and final_reply.text.strip():
            send_result = self.sender_service.send_text_reply(
                conversation_id,
                final_reply.text,
                final_reply.delay_seconds,
                raw_payload=inbound_message.raw_payload,
                metadata={
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "delivery": "queued",
                },
            )
            self.conversation_repo.save_message(
                conversation_id,
                "ai",
                "text",
                final_reply.text,
                {
                    "selected_content": selected_content,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "delivery": send_result,
                },
                None,
            )
            self.conversation_repo.set_last_ai_reply_at(conversation_id)
        self.conversation_repo.update_conversation_state(conversation_id, stage_decision.stage, mode_decision.chat_mode, understanding.current_mainline_should_continue)
        if project_decision.project_id is not None:
            self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
        self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_decision["ops_category"], ops_decision["reason"], "system")
        try:
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "AI Turn Decision",
                {
                    "conversation_id": conversation_id,
                    "stage": stage_decision.stage,
                    "stage_confidence": stage_decision.confidence,
                    "chat_mode": mode_decision.chat_mode,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "intent_level": intent_decision.level,
                    "intent_score": intent_decision.score,
                },
            )
        except Exception:
            logger.exception("Failed to create AI turn-decision receipt")
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
                    "style_spec": asdict(style_spec),
                },
            )
            if self.admin_notifier:
                self.admin_notifier.notify_high_intent(business_account_id, user_id, conversation_id, escalation_decision.get("notify_level") or "watch", escalation_decision.get("reason") or "")


class AsyncSenderWorker:
    def run_forever(self) -> None:
        db = Database(self.database_url)
        db.connect()
        repo = OutboundMessageJobRepository(db)
        while not self._stop_event.is_set():
            try:
                jobs = repo.fetch_ready_jobs(limit=10)
                if not jobs:
                    time.sleep(self.poll_interval)
                    continue
                for job in jobs:
                    job_id = int(job["id"])
                    payload = job.get("payload_json") or {}
                    try:
                        repo.mark_sending(job_id)
                        method = "sendMessage"
                        api_payload = {
                            "chat_id": payload.get("chat_id"),
                            "text": payload.get("text") or "",
                            "business_connection_id": payload.get("business_connection_id"),
                        }
                        if not api_payload.get("chat_id") or not api_payload.get("text"):
                            raise RuntimeError("missing delivery context for outbound reply job")
                        self._call_api(method, api_payload)
                        repo.mark_sent(job_id)
                    except Exception as exc:
                        logger.exception("Async sender worker failed job_id=%s", job_id)
                        repo.mark_failed(job_id, str(exc))
                time.sleep(0.1)
            except Exception:
                logger.exception("Async sender loop error")
                time.sleep(self.poll_interval)
        db.close()

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


# =========================
# V1 upgrade layer
# =========================
import threading
from dataclasses import asdict


@dataclass
class MemoryBundle:
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    recent_summary: str = ""
    long_term_memory: dict[str, Any] = field(default_factory=dict)
    handover_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnDecision:
    reply_goal: str = "rapport"
    should_push_project: bool = False
    social_distance: str = "balanced"
    reply_length: str = "medium"
    ask_followup_question: bool = False
    exit_strategy: str = "gentle_close"
    self_disclosure_level: str = "light"
    reason: str = ""


class ProcessedUpdateRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def exists(self, update_key: str) -> bool:
        with self.db.cursor() as cur:
            cur.execute("SELECT 1 FROM processed_updates WHERE update_key=%s LIMIT 1", (update_key,))
            return cur.fetchone() is not None

    def insert(self, update_key: str, update_type: str, telegram_update_id: str | None, business_connection_id: str | None, telegram_message_id: int | None) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO processed_updates (update_key, update_type, telegram_update_id, business_connection_id, telegram_message_id)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (update_key) DO NOTHING
                    """,
                    (update_key, update_type, telegram_update_id, business_connection_id, telegram_message_id),
                )


class OutboundMessageJobRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_job(self, conversation_id: int, payload: dict[str, Any], send_after_seconds: int = 0, job_type: str = "reply") -> int:
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO outbound_message_jobs (conversation_id, job_type, payload_json, send_after, status, retry_count)
                    VALUES (%s,%s,%s::jsonb,NOW() + (%s * INTERVAL '1 second'),'pending',0)
                    RETURNING id
                    """,
                    (conversation_id, job_type, payload_json, max(int(send_after_seconds or 0), 0)),
                )
                return int(cur.fetchone()["id"])

    def fetch_ready_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT id, conversation_id, job_type, payload_json, retry_count
                FROM outbound_message_jobs
                WHERE status='pending' AND send_after <= NOW()
                ORDER BY send_after ASC, id ASC
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())

    def mark_sending(self, job_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE outbound_message_jobs SET status='sending', updated_at=NOW() WHERE id=%s", (job_id,))

    def mark_sent(self, job_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE outbound_message_jobs SET status='sent', sent_at=NOW(), updated_at=NOW(), last_error=NULL WHERE id=%s", (job_id,))

    def mark_failed(self, job_id: int, error: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE outbound_message_jobs
                    SET status=CASE WHEN retry_count >= 4 THEN 'failed' ELSE 'pending' END,
                        retry_count=retry_count+1,
                        last_error=%s,
                        send_after=CASE WHEN retry_count >= 4 THEN send_after ELSE NOW() + ((retry_count + 1) * INTERVAL '15 second') END,
                        updated_at=NOW()
                    WHERE id=%s
                    """,
                    (error[:1000], job_id),
                )


class WebhookIdempotencyGuard:
    def __init__(self, repo: ProcessedUpdateRepository) -> None:
        self.repo = repo

    def _build_update_key(self, raw_update: dict[str, Any]) -> tuple[str, str, str | None, int | None]:
        update_type = "unknown"
        if raw_update.get("business_message"):
            update_type = "business_message"
        elif raw_update.get("edited_business_message"):
            update_type = "edited_business_message"
        elif raw_update.get("callback_query"):
            update_type = "callback_query"
        elif raw_update.get("message"):
            update_type = "message"
        tg_update_id = raw_update.get("update_id")
        business_connection_id = _extract_business_connection_id(raw_update)
        telegram_message_id = _extract_message_id(raw_update)
        if tg_update_id is not None:
            return (f"tg_update:{tg_update_id}", update_type, str(tg_update_id), telegram_message_id)
        return (f"composite:{update_type}:{business_connection_id or '-'}:{telegram_message_id or 0}", update_type, None, telegram_message_id)

    def check_and_mark(self, raw_update: dict[str, Any]) -> tuple[bool, str]:
        update_key, update_type, tg_update_id, telegram_message_id = self._build_update_key(raw_update)
        if self.repo.exists(update_key):
            return False, update_key
        self.repo.insert(update_key, update_type, tg_update_id, _extract_business_connection_id(raw_update), telegram_message_id)
        return True, update_key


class OutboundMessageQueue:
    def __init__(self, repo: OutboundMessageJobRepository, tg_client: TelegramBotAPIClient) -> None:
        self.repo = repo
        self.tg_client = tg_client

    def enqueue_text_reply(self, conversation_id: int, text: str, raw_payload: dict[str, Any] | None = None, delay_seconds: int = 0, metadata: dict[str, Any] | None = None) -> int:
        payload = dict(metadata or {})
        payload.update({
            "text": text,
            "chat_id": _extract_chat_id(raw_payload or {}),
            "business_connection_id": _extract_business_connection_id(raw_payload or {}),
        })
        if payload.get("chat_id") is None:
            delivery = self.tg_client._get_delivery_context_for_conversation(conversation_id)
            payload["chat_id"] = delivery.get("chat_id")
            payload["business_connection_id"] = delivery.get("business_connection_id")
        return self.repo.insert_job(conversation_id, payload, delay_seconds, job_type="reply")


_Prev_TurnDecisionEngine_1 = TurnDecisionEngine
class TurnDecisionEngine(_Prev_TurnDecisionEngine_1):
    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        if understanding.boundary_signal:
            return TurnDecision(
                reply_goal="respect_boundary",
                should_push_project=False,
                social_distance="give_space",
                reply_length="short",
                ask_followup_question=False,
                exit_strategy="leave_space",
                self_disclosure_level="none",
                reason=f"boundary:{understanding.boundary_signal}",
            )
        if understanding.high_intent_signal:
            return TurnDecision(
                reply_goal="progress_without_pressure",
                should_push_project=True,
                social_distance="professional_close",
                reply_length="medium",
                ask_followup_question=True,
                exit_strategy="soft_next_step",
                self_disclosure_level="none",
                reason="high_intent_signal",
            )
        if understanding.explicit_product_query or understanding.product_interest_signal:
            return TurnDecision(
                reply_goal="clarify_project_naturally",
                should_push_project=True,
                social_distance="balanced",
                reply_length="medium",
                ask_followup_question=False,
                exit_strategy="soft_open_end",
                self_disclosure_level="light" if "product_interest" not in (user_state.tags or []) else "none",
                reason="product_interest_or_query",
            )
        if understanding.emotion_state in ("low", "anxious"):
            return TurnDecision(
                reply_goal="emotional_support_first",
                should_push_project=False,
                social_distance="warm",
                reply_length="medium",
                ask_followup_question=False,
                exit_strategy="gentle_close",
                self_disclosure_level="none",
                reason="emotion_support_needed",
            )
        return TurnDecision(
            reply_goal="build_rapport",
            should_push_project=False,
            social_distance="balanced",
            reply_length="medium",
            ask_followup_question=False,
            exit_strategy="gentle_close",
            self_disclosure_level="light",
            reason=f"default_stage:{stage_decision.stage}|mode:{mode_decision.chat_mode}",
        )


_Prev_TelegramUpdateMapper_1 = TelegramUpdateMapper
class TelegramUpdateMapper(_Prev_TelegramUpdateMapper_1):
    def map_update(self, raw_update: dict) -> InboundMessage | None:
        message_obj = None
        update_type = None
        if raw_update.get("business_message"):
            message_obj = raw_update["business_message"]
            update_type = "business_message"
        elif raw_update.get("edited_business_message"):
            message_obj = raw_update["edited_business_message"]
            update_type = "edited_business_message"
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
        normalized_payload["update_type"] = update_type
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


_Prev_SenderService_1 = SenderService
class SenderService(_Prev_SenderService_1):
    def __init__(self, sender_adapter: TelegramBusinessSenderAdapter, outbound_queue: OutboundMessageQueue | None = None) -> None:
        super().__init__(sender_adapter)
        self.outbound_queue = outbound_queue

    def send_text_reply(self, conversation_id: int, text: str, delay_seconds: int = 0, raw_payload: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.outbound_queue is not None:
            job_id = self.outbound_queue.enqueue_text_reply(
                conversation_id=conversation_id,
                text=text,
                raw_payload=raw_payload,
                delay_seconds=delay_seconds,
                metadata=metadata,
            )
            return {"queued": True, "job_id": job_id}
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        self.sender_adapter.send_text(conversation_id, text)
        return {"queued": False, "job_id": None}


_Prev_ReplySelfCheckEngine_2 = ReplySelfCheckEngine
class ReplySelfCheckEngine(_Prev_ReplySelfCheckEngine_2):
    def check_and_fix(self, draft_reply: str, mode: str, understanding: UnderstandingResult) -> str:
        text = super().check_and_fix(draft_reply, mode, understanding)
        banned_fragments = [
            "as an ai", "i am an ai", "customer support", "our platform team", "kindly note",
            "dear user", "please let me know if you have any further questions"
        ]
        lowered = text.lower()
        for frag in banned_fragments:
            if frag in lowered:
                text = text.replace(frag, "")
                text = text.replace(frag.title(), "")
        text = " ".join(text.split())
        if understanding.boundary_signal and text.endswith("?"):
            text = text[:-1].rstrip() + "."
        return text.strip()


_Prev_ReplyDelayEngine_2 = ReplyDelayEngine
class ReplyDelayEngine(_Prev_ReplyDelayEngine_2):
    def decide_delay_seconds(self, understanding: UnderstandingResult, mode: str, reply_text: str) -> int:
        if understanding.boundary_signal:
            return 1
        if mode == "high_intent":
            return 2
        if mode == "product":
            return 4
        if mode == "emotional":
            return 3
        return 2


_Prev_Orchestrator_2 = Orchestrator
class Orchestrator(_Prev_Orchestrator_2):
    def __init__(self, *args, turn_decision_engine: TurnDecisionEngine | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.turn_decision_engine = turn_decision_engine or TurnDecisionEngine()

    def _build_memory_bundle(self, context: ConversationContext, latest_handover_summary: dict[str, Any] | None = None) -> MemoryBundle:
        return MemoryBundle(
            recent_messages=list(context.recent_messages or []),
            recent_summary=context.recent_summary or "",
            long_term_memory=context.long_term_memory or {},
            handover_summary=latest_handover_summary or {},
        )

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

        latest_user_text = (inbound_message.text or "").strip()
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
        memory_bundle = self._build_memory_bundle(context, latest_handover_summary)
        understanding = self.understanding_engine.analyze(latest_user_text, recent_context, self.persona_core.to_summary(), self._user_state_summary(user_state))
        stage_decision = self.stage_engine.decide(understanding, context.current_stage)
        mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
        turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
        project_decision = self.project_classifier.classify(business_account_id, understanding, context, user_state, latest_user_text)
        intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
        segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
        escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
        tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
        ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)
        reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
        reply_plan.should_continue_product = turn_decision.should_push_project
        reply_plan.should_leave_space = turn_decision.exit_strategy == "leave_space"
        reply_plan.should_self_share = turn_decision.self_disclosure_level != "none"
        selected_content = self.content_selector.select(business_account_id, project_decision.project_id, mode_decision.chat_mode, reply_plan)
        understanding_payload = understanding.__dict__.copy()
        understanding_payload["turn_decision"] = asdict(turn_decision)
        understanding_payload["memory_summary"] = {
            "recent_messages_count": len(memory_bundle.recent_messages),
            "has_recent_summary": bool(memory_bundle.recent_summary),
            "has_handover_summary": bool(memory_bundle.handover_summary),
        }
        if latest_handover_summary:
            understanding_payload["resume_hint"] = latest_handover_summary.get("resume_suggestion")
        draft_reply = self.reply_style_engine.generate(
            latest_user_text,
            recent_context,
            persona_summary,
            self._user_state_summary(user_state),
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding_payload,
            reply_plan.__dict__,
            selected_content,
        )
        final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        final_reply = FinalReply(text=final_text, delay_seconds=delay_seconds, metadata={"turn_decision": asdict(turn_decision)})
        if reply_plan.should_reply and final_reply.text.strip():
            send_result = self.sender_service.send_text_reply(
                conversation_id,
                final_reply.text,
                final_reply.delay_seconds,
                raw_payload=inbound_message.raw_payload,
                metadata={"turn_decision": asdict(turn_decision), "delivery": "queued"},
            )
            self.conversation_repo.save_message(
                conversation_id,
                "ai",
                "text",
                final_reply.text,
                {
                    "selected_content": selected_content,
                    "turn_decision": asdict(turn_decision),
                    "delivery": send_result,
                },
                None,
            )
            self.conversation_repo.set_last_ai_reply_at(conversation_id)
        self.conversation_repo.update_conversation_state(conversation_id, stage_decision.stage, mode_decision.chat_mode, understanding.current_mainline_should_continue)
        if project_decision.project_id is not None:
            self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
        self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_decision["ops_category"], ops_decision["reason"], "system")
        try:
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "AI Turn Decision",
                {
                    "conversation_id": conversation_id,
                    "stage": stage_decision.stage,
                    "chat_mode": mode_decision.chat_mode,
                    "turn_decision": asdict(turn_decision),
                    "intent_level": intent_decision.level,
                    "intent_score": intent_decision.score,
                },
            )
        except Exception:
            logger.exception("Failed to create AI turn-decision receipt")
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


_Prev_AsyncSenderWorker_1 = AsyncSenderWorker
class AsyncSenderWorker(_Prev_AsyncSenderWorker_1):
    def __init__(self, database_url: str, bot_token: str, poll_interval: float = 1.0) -> None:
        self.database_url = database_url
        self.bot_token = bot_token
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run_forever, name="async-sender-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _call_api(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        base_url = f"https://api.telegram.org/bot{self.bot_token}"
        clean_payload = {k: v for k, v in payload.items() if v is not None}
        req = urllib_request.Request(
            url=f"{base_url}/{method}",
            data=json.dumps(clean_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        if not parsed.get("ok"):
            raise RuntimeError(f"Telegram API call failed on {method}: {parsed}")
        return parsed

    def run_forever(self) -> None:
        db = Database(self.database_url)
        db.connect()
        repo = OutboundMessageJobRepository(db)
        while not self._stop_event.is_set():
            try:
                jobs = repo.fetch_ready_jobs(limit=10)
                if not jobs:
                    time.sleep(self.poll_interval)
                    continue
                for job in jobs:
                    job_id = int(job["id"])
                    payload = job.get("payload_json") or {}
                    try:
                        repo.mark_sending(job_id)
                        self._call_api(
                            "sendMessage",
                            {
                                "chat_id": payload.get("chat_id"),
                                "text": payload.get("text") or "",
                                "business_connection_id": payload.get("business_connection_id"),
                            },
                        )
                        repo.mark_sent(job_id)
                    except Exception as exc:
                        logger.exception("Async sender worker failed | job_id=%s", job_id)
                        repo.mark_failed(job_id, str(exc))
            except Exception:
                logger.exception("Async sender worker loop failed")
                time.sleep(max(self.poll_interval, 2.0))


def ensure_v1_upgrade_schema(db: Database) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS processed_updates (
            id BIGSERIAL PRIMARY KEY,
            update_key TEXT NOT NULL UNIQUE,
            update_type TEXT NOT NULL,
            telegram_update_id TEXT,
            business_connection_id TEXT,
            telegram_message_id BIGINT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS outbound_message_jobs (
            id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            job_type TEXT NOT NULL,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            send_after TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status TEXT NOT NULL DEFAULT 'pending',
            retry_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            sent_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ]
    with db.transaction():
        with db.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_processed_updates_created_at ON processed_updates(created_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_outbound_message_jobs_ready ON outbound_message_jobs(status, send_after)")


def create_web_app(settings: Settings, app_components: dict[str, Any]) -> Flask:
    flask_app = Flask(__name__)
    admin_api_service = app_components["admin_api_service"]
    dashboard_service = app_components["dashboard_service"]
    gateway = app_components["gateway"]
    tg_admin_handlers = app_components["tg_admin_handlers"]
    tg_client = app_components["tg_client"]
    idempotency_guard: WebhookIdempotencyGuard = app_components["idempotency_guard"]
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

            if raw_update.get("business_message") or raw_update.get("edited_business_message"):
                should_continue, update_key = idempotency_guard.check_and_mark(raw_update)
                if not should_continue:
                    return jsonify({"ok": True, "dispatched": "duplicate_ignored", "update_key": update_key})
                if raw_update.get("edited_business_message"):
                    return jsonify({"ok": True, "dispatched": "edited_ignored", "update_key": update_key})

            dispatched = gateway.handle_raw_update(raw_update)
            return jsonify({"ok": True, "dispatched": "business_message" if dispatched else "ignored"})
        except Exception as e:
            try:
                app_components["db"].rollback()
            except Exception:
                pass
            logger.exception("telegram_webhook failed")
            return jsonify({"ok": False, "error": str(e)}), 500

    return flask_app


def build_app_components(settings: Settings) -> dict[str, Any]:
    db = Database(settings.database_url)
    db.connect()
    initialize_database(db)
    ensure_v1_upgrade_schema(db)

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
    processed_update_repo = ProcessedUpdateRepository(db)
    outbound_message_job_repo = OutboundMessageJobRepository(db)

    openai_adapter = OpenAIClientAdapter(settings.openai_api_key)
    llm_service = LLMService(openai_adapter, settings.llm_model_name)
    outbound_queue = OutboundMessageQueue(outbound_message_job_repo, tg_client)
    sender_service = SenderService(TelegramBusinessSenderAdapter(tg_client), outbound_queue=outbound_queue)
    admin_notifier = AdminNotifier(tg_client, admin_chat_ids=settings.admin_chat_ids)
    idempotency_guard = WebhookIdempotencyGuard(processed_update_repo)
    sender_worker = AsyncSenderWorker(settings.database_url, settings.tg_bot_token)

    persona_core = PersonaCore()
    persona_profile_builder = PersonaProfileBuilder(material_repo)
    understanding_engine = UserUnderstandingEngine(llm_service)
    stage_engine = ConversationStageEngine()
    mode_router = ChatModeRouter()
    reply_planner = ReplyPlanner()
    reply_style_engine = ReplyStyleEngine(llm_service)
    reply_self_check_engine = ReplySelfCheckEngine()
    reply_delay_engine = ReplyDelayEngine()
    turn_decision_engine = TurnDecisionEngine()

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
        turn_decision_engine=turn_decision_engine,
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
        "idempotency_guard": idempotency_guard,
        "sender_worker": sender_worker,
    }



# =========================
# V2 upgrade layer
# =========================

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
    boundary_sensitivity: float = 0.25
    response_speed_preference: str = "normal"
    emotional_receptiveness: float = 0.5
    professional_receptiveness: float = 0.5
    recent_busy_score: float = 0.0
    push_fatigue_score: float = 0.0
    openness_score: float = 0.45
    warmth_score: float = 0.45
    current_project_interest_strength: float = 0.0


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
    emotion_strength: float = 0.4
    boundary_strength: float = 0.0
    social_openness: float = 0.45
    project_relevance: float = 0.0
    human_takeover_hint: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class StageDecision:
    stage: str
    changed: bool
    reason: str
    confidence: float = 0.65


@dataclass
class TurnDecision:
    reply_goal: str = "rapport"
    should_push_project: bool = False
    social_distance: str = "balanced"
    reply_length: str = "medium"
    ask_followup_question: bool = False
    exit_strategy: str = "gentle_close"
    self_disclosure_level: str = "light"
    reason: str = ""
    should_reply: bool = True
    marketing_intensity: str = "subtle"
    tone_bias: str = "natural"


@dataclass
class StyleSpec:
    tone: str = "natural"
    formality: str = "balanced"
    warmth: str = "medium"
    length: str = "medium"
    question_rate: str = "low"
    emoji_usage: str = "minimal"
    self_disclosure_ratio: str = "none"
    completion_level: str = "balanced"
    marketing_visibility: str = "subtle"
    naturalness_bias: str = "high"


_Prev_UserRepository_2 = UserRepository
class UserRepository(_Prev_UserRepository_2):
    def get_user_state_snapshot(self, business_account_id: int, user_id: int) -> UserStateSnapshot:
        base = super().get_user_state_snapshot(business_account_id, user_id)
        tags = set(base.tags or [])

        relationship_score = 0.28
        trust_score = 0.24
        comfort_score = 0.34
        current_heat = 0.26
        marketing_tolerance = 0.42
        boundary_sensitivity = 0.28
        response_speed_preference = "normal"
        emotional_receptiveness = 0.52
        professional_receptiveness = 0.52
        recent_busy_score = 0.0
        push_fatigue_score = 0.12
        openness_score = 0.46
        warmth_score = 0.48
        current_project_interest_strength = 0.0

        if "recently_busy" in tags:
            recent_busy_score = 0.88
            boundary_sensitivity = 0.82
            response_speed_preference = "slow"
            marketing_tolerance = min(marketing_tolerance, 0.25)
            push_fatigue_score = max(push_fatigue_score, 0.62)
        if "cautious" in tags:
            trust_score = 0.38
            professional_receptiveness = 0.65
            emotional_receptiveness = 0.45
            marketing_tolerance = min(marketing_tolerance, 0.33)
            boundary_sensitivity = max(boundary_sensitivity, 0.52)
        if "product_interest" in tags:
            current_project_interest_strength = 0.68
            professional_receptiveness = 0.74
            relationship_score = max(relationship_score, 0.44)
            current_heat = max(current_heat, 0.48)
        if "followup_worthy" in tags:
            current_heat = max(current_heat, 0.54)
            openness_score = max(openness_score, 0.58)
        if "high_intent" in tags:
            current_project_interest_strength = 0.9
            trust_score = max(trust_score, 0.7)
            relationship_score = max(relationship_score, 0.66)
            current_heat = max(current_heat, 0.74)
            marketing_tolerance = max(marketing_tolerance, 0.66)
            professional_receptiveness = max(professional_receptiveness, 0.82)
            openness_score = max(openness_score, 0.7)
            warmth_score = max(warmth_score, 0.62)

        if base.ops_category in ("intent_user", "deal_user"):
            relationship_score = max(relationship_score, 0.58)
            trust_score = max(trust_score, 0.6)
            current_heat = max(current_heat, 0.66)
            marketing_tolerance = max(marketing_tolerance, 0.62)

        return UserStateSnapshot(
            ops_category=base.ops_category,
            project_id=base.project_id,
            project_segment_id=base.project_segment_id,
            tags=list(base.tags or []),
            relationship_score=round(relationship_score, 2),
            trust_score=round(trust_score, 2),
            comfort_score=round(comfort_score, 2),
            current_heat=round(current_heat, 2),
            marketing_tolerance=round(marketing_tolerance, 2),
            boundary_sensitivity=round(boundary_sensitivity, 2),
            response_speed_preference=response_speed_preference,
            emotional_receptiveness=round(emotional_receptiveness, 2),
            professional_receptiveness=round(professional_receptiveness, 2),
            recent_busy_score=round(recent_busy_score, 2),
            push_fatigue_score=round(push_fatigue_score, 2),
            openness_score=round(openness_score, 2),
            warmth_score=round(warmth_score, 2),
            current_project_interest_strength=round(current_project_interest_strength, 2),
        )


_Prev_UserUnderstandingEngine_2 = UserUnderstandingEngine
class UserUnderstandingEngine(_Prev_UserUnderstandingEngine_2):
    def analyze(self, latest_user_message: str, recent_context: list[dict], persona_core_summary: str, user_state_summary: str) -> UnderstandingResult:
        fallback = self._fallback_rule_based(latest_user_message)
        try:
            result = self.llm_service.classify_user_message(
                build_understanding_prompt(latest_user_message, recent_context, persona_core_summary, user_state_summary)
            )
        except Exception:
            result = fallback
        merged = dict(fallback)
        for k, v in (result or {}).items():
            if v is not None:
                merged[k] = v
        return UnderstandingResult(**merged)

    def _fallback_rule_based(self, text: str) -> dict:
        t = (text or "").strip().lower()
        busy_words = ["busy", "later", "sleep", "sleeping", "work", "working", "tomorrow", "meeting", "driving"]
        emotional_words = ["tired", "sad", "stress", "stressed", "anxious", "worried", "upset", "rough day"]
        product_words = ["return", "profit", "plan", "project", "rate", "yield", "details", "how does it work", "risk"]
        high_intent_words = ["interested", "want to know", "can i join", "how do i start", "what do i need", "send me details"]
        boundary_signal = None
        boundary_strength = 0.0
        emotion_state = "neutral"
        emotion_strength = 0.35
        if any(w in t for w in busy_words):
            boundary_signal = "busy"
            boundary_strength = 0.9
        elif any(w in t for w in ["not now", "stop", "later please", "leave it", "another time"]):
            boundary_signal = "needs_space"
            boundary_strength = 0.95
        if any(w in t for w in emotional_words):
            emotion_state = "low"
            emotion_strength = 0.76
        elif any(w in t for w in ["good", "nice", "great", "sounds good", "interesting"]):
            emotion_state = "positive"
            emotion_strength = 0.62
        product_interest = any(w in t for w in product_words)
        explicit_product_query = product_interest or "?" in t and any(w in t for w in ["plan", "return", "rate", "risk", "how"])
        high_intent_signal = any(w in t for w in high_intent_words)
        social_openness = 0.38
        if any(w in t for w in ["haha", "lol", "how about you", "what about you", "really", "yeah"]):
            social_openness = 0.62
        if boundary_signal:
            social_openness = min(social_openness, 0.28)
        project_relevance = 0.8 if explicit_product_query else (0.55 if product_interest else 0.15)
        human_takeover_hint = bool(high_intent_signal and any(w in t for w in ["call", "agent", "real person", "someone can explain"]))
        recommended_chat_mode = "pause" if boundary_signal else ("high_intent" if high_intent_signal else ("product" if explicit_product_query else ("emotional" if emotion_state == "low" else "daily")))
        recommended_goal = (
            "respect_boundary" if boundary_signal else
            "progress_without_pressure" if high_intent_signal else
            "answer_and_transition" if explicit_product_query else
            "support_first" if emotion_state == "low" else
            "build_rapport"
        )
        return {
            "user_intent": "product_query" if explicit_product_query else ("interest" if high_intent_signal else "chat"),
            "need_type": "information" if explicit_product_query else ("progress" if high_intent_signal else "rapport"),
            "emotion_state": emotion_state,
            "boundary_signal": boundary_signal,
            "resistance_signal": "soft_resistance" if "not sure" in t or "maybe" in t else None,
            "product_interest_signal": product_interest,
            "explicit_product_query": explicit_product_query,
            "high_intent_signal": high_intent_signal,
            "current_mainline_should_continue": "product" if (explicit_product_query or high_intent_signal) else "daily",
            "recommended_chat_mode": recommended_chat_mode,
            "recommended_goal": recommended_goal,
            "reason": "v2_rule_fallback",
            "emotion_strength": emotion_strength,
            "boundary_strength": boundary_strength,
            "social_openness": social_openness,
            "project_relevance": project_relevance,
            "human_takeover_hint": human_takeover_hint,
            "notes": [],
        }


_Prev_ConversationStageEngine_2 = ConversationStageEngine
class ConversationStageEngine(_Prev_ConversationStageEngine_2):
    def decide(self, understanding: UnderstandingResult, current_stage: str | None) -> StageDecision:
        target = current_stage or "first_contact"
        reason = "retain current stage"
        confidence = 0.66
        if understanding.boundary_signal:
            target = "boundary_protection"
            reason = f"boundary detected: {understanding.boundary_signal}"
            confidence = max(0.82, understanding.boundary_strength)
        elif understanding.high_intent_signal:
            target = "high_intent_progress"
            reason = "high intent detected"
            confidence = 0.88
        elif understanding.explicit_product_query or understanding.project_relevance >= 0.75:
            target = "project_discussion"
            reason = "project discussion detected"
            confidence = 0.79
        elif current_stage in (None, "", "new", "relationship", "first_contact"):
            if understanding.social_openness >= 0.58 or understanding.emotion_state in ("positive", "low"):
                target = "trust_building"
                reason = "rapport or emotional openness detected"
                confidence = 0.71
            else:
                target = "daily_rapport"
                reason = "light rapport maintenance"
                confidence = 0.64
        return StageDecision(stage=target, changed=(current_stage != target), reason=reason, confidence=confidence)


_Prev_TurnDecisionEngine_2 = TurnDecisionEngine
class TurnDecisionEngine(_Prev_TurnDecisionEngine_2):
    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        if understanding.boundary_signal:
            return TurnDecision(
                reply_goal="respect_boundary",
                should_push_project=False,
                social_distance="give_space",
                reply_length="short",
                ask_followup_question=False,
                exit_strategy="leave_space",
                self_disclosure_level="none",
                reason=f"boundary:{understanding.boundary_signal}",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="gentle",
            )
        if understanding.high_intent_signal:
            ask_followup = user_state.boundary_sensitivity < 0.65 and user_state.push_fatigue_score < 0.7
            return TurnDecision(
                reply_goal="progress_without_pressure",
                should_push_project=True,
                social_distance="professional_close",
                reply_length="medium",
                ask_followup_question=ask_followup,
                exit_strategy="soft_next_step",
                self_disclosure_level="none",
                reason="high_intent_signal",
                should_reply=True,
                marketing_intensity="light",
                tone_bias="professional_warm",
            )
        if understanding.explicit_product_query or understanding.project_relevance >= 0.72:
            return TurnDecision(
                reply_goal="answer_and_transition",
                should_push_project=True,
                social_distance="balanced",
                reply_length="medium",
                ask_followup_question=user_state.boundary_sensitivity < 0.55,
                exit_strategy="gentle_close",
                self_disclosure_level="none",
                reason="product_focus",
                should_reply=True,
                marketing_intensity="subtle",
                tone_bias="clear",
            )
        if understanding.emotion_state == "low":
            return TurnDecision(
                reply_goal="support_first",
                should_push_project=False,
                social_distance="warm_supportive",
                reply_length="short",
                ask_followup_question=False,
                exit_strategy="soft_hold",
                self_disclosure_level="none",
                reason="emotion_low",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="soft",
            )
        if stage_decision.stage in ("trust_building", "daily_rapport"):
            return TurnDecision(
                reply_goal="build_rapport",
                should_push_project=False,
                social_distance="friendly_but_measured",
                reply_length="short" if user_state.response_speed_preference == "slow" else "medium",
                ask_followup_question=user_state.boundary_sensitivity < 0.45,
                exit_strategy="light_hook" if user_state.boundary_sensitivity < 0.45 else "gentle_close",
                self_disclosure_level="light" if understanding.social_openness >= 0.55 else "none",
                reason="rapport_stage",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="natural",
            )
        return TurnDecision(
            reply_goal="maintain_presence",
            should_push_project=False,
            social_distance="balanced",
            reply_length="medium",
            ask_followup_question=False,
            exit_strategy="gentle_close",
            self_disclosure_level="none",
            reason="default_v2",
            should_reply=True,
            marketing_intensity="none" if user_state.boundary_sensitivity > 0.55 else "subtle",
            tone_bias="natural",
        )


_Prev_HumanizationController_1 = HumanizationController
class HumanizationController(_Prev_HumanizationController_1):
    def build_style_spec(
        self,
        decision: TurnDecision,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        persona_profile: dict[str, Any] | None = None,
    ) -> StyleSpec:
        _ = persona_profile
        tone = "natural"
        formality = "balanced"
        warmth = "medium"
        question_rate = "low"
        completion_level = "balanced"
        marketing_visibility = decision.marketing_intensity
        self_disclosure_ratio = decision.self_disclosure_level
        if decision.reply_goal == "respect_boundary":
            tone = "gentle"
            formality = "casual"
            warmth = "soft"
            question_rate = "none"
            completion_level = "lean"
        elif decision.reply_goal == "support_first":
            tone = "soft"
            formality = "casual"
            warmth = "high"
            question_rate = "none"
            completion_level = "lean"
        elif decision.reply_goal in ("answer_and_transition", "progress_without_pressure"):
            tone = "clear"
            formality = "balanced"
            warmth = "medium"
            question_rate = "one"
            completion_level = "focused"
        elif stage_decision.stage in ("trust_building", "daily_rapport"):
            tone = "natural"
            formality = "casual"
            warmth = "medium_high"
            question_rate = "low" if not decision.ask_followup_question else "one"
            completion_level = "light"
        if user_state.response_speed_preference == "slow":
            completion_level = "lean"
        if user_state.boundary_sensitivity > 0.7:
            question_rate = "none"
            marketing_visibility = "none"
        return StyleSpec(
            tone=tone,
            formality=formality,
            warmth=warmth,
            length=decision.reply_length,
            question_rate=question_rate,
            emoji_usage="minimal",
            self_disclosure_ratio=self_disclosure_ratio,
            completion_level=completion_level,
            marketing_visibility=marketing_visibility,
            naturalness_bias="high",
        )


_Prev_ReplyStyleEngine_2 = ReplyStyleEngine
class ReplyStyleEngine(_Prev_ReplyStyleEngine_2):
    def generate(self, latest_user_message: str, recent_context: list[dict], persona_summary: str, user_state_summary: str, stage: str, chat_mode: str, understanding: dict, reply_plan: dict, selected_content: dict) -> str:
        text = super().generate(latest_user_message, recent_context, persona_summary, user_state_summary, stage, chat_mode, understanding, reply_plan, selected_content)
        style_spec = understanding.get("style_spec") or {}
        turn_decision = understanding.get("turn_decision") or {}
        return self._apply_style_postprocess(text, style_spec, turn_decision)

    def _apply_style_postprocess(self, text: str, style_spec: dict[str, Any], turn_decision: dict[str, Any]) -> str:
        t = " ".join((text or "").split()).strip()
        if not t:
            return "I’m here."
        length = style_spec.get("length") or "medium"
        completion_level = style_spec.get("completion_level") or "balanced"
        max_len = 340
        if length == "short":
            max_len = 170
        elif length == "medium":
            max_len = 320
        else:
            max_len = 520
        if completion_level == "lean":
            max_len = min(max_len, 180)
        marketing_visibility = style_spec.get("marketing_visibility") or "none"
        if marketing_visibility == "none":
            for frag in ["join now", "start now", "we can get you started", "this is a good time to enter", "profit", "returns"]:
                t = t.replace(frag, "")
                t = t.replace(frag.title(), "")
        question_rate = style_spec.get("question_rate") or "low"
        if question_rate == "none":
            if "?" in t:
                t = t.replace("?", ".")
        elif question_rate == "one":
            first = t.find("?")
            if first != -1:
                later = t.find("?", first + 1)
                if later != -1:
                    t = t[:later].replace("?", ".") + t[later:]
                    t = t.replace("?", ".")
                    if first != -1:
                        lst = list(t)
                        lst[first] = "?"
                        t = "".join(lst)
        if turn_decision.get("exit_strategy") in ("leave_space", "soft_hold") and t.endswith("?"):
            t = t[:-1].rstrip() + "."
        if len(t) > max_len:
            cut = t[:max_len].rstrip()
            if "." in cut[-60:]:
                cut = cut[: cut.rfind(".") + 1]
            else:
                cut += "..."
            t = cut.strip()
        return " ".join(t.split()).strip()


_Prev_ReplySelfCheckEngine_3 = ReplySelfCheckEngine
class ReplySelfCheckEngine(_Prev_ReplySelfCheckEngine_3):
    def check_and_fix(self, draft_reply: str, mode: str, understanding: UnderstandingResult) -> str:
        text = super().check_and_fix(draft_reply, mode, understanding)
        lowered = text.lower()
        banned_fragments = [
            "as an ai", "i am an ai", "customer support", "our platform team", "kindly note",
            "dear user", "please let me know if you have any further questions", "allow me to explain"
        ]
        for frag in banned_fragments:
            if frag in lowered:
                text = text.replace(frag, "")
                text = text.replace(frag.title(), "")
                lowered = text.lower()
        if understanding.boundary_signal:
            for frag in ["can i", "would you like", "shall i", "do you want"]:
                if frag in lowered:
                    text = text.replace(frag, "")
                    lowered = text.lower()
            text = text.replace("??", "?")
            if text.endswith("?"):
                text = text[:-1].rstrip() + "."
        if understanding.emotion_state == "low" and not understanding.explicit_product_query:
            for risky in ["returns", "profit", "join now", "start now", "earn"]:
                text = text.replace(risky, "")
                text = text.replace(risky.title(), "")
        text = " ".join(text.split()).strip()
        if len(text) > 420:
            text = text[:420].rstrip() + "..."
        return text.strip()


_Prev_ReplyDelayEngine_3 = ReplyDelayEngine
class ReplyDelayEngine(_Prev_ReplyDelayEngine_3):
    def decide_delay_seconds(self, understanding: UnderstandingResult, mode: str, reply_text: str) -> int:
        _ = reply_text
        if understanding.boundary_signal:
            return 1
        if mode == "high_intent":
            return 2
        if mode == "product":
            return 3
        if mode == "emotional":
            return 2
        if understanding.social_openness >= 0.58:
            return 2
        return 3


_Prev_Orchestrator_3 = Orchestrator
class Orchestrator(_Prev_Orchestrator_3):
    def __init__(self, *args, turn_decision_engine: TurnDecisionEngine | None = None, humanization_controller: HumanizationController | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.turn_decision_engine = turn_decision_engine or TurnDecisionEngine()
        self.humanization_controller = humanization_controller or HumanizationController()

    def _build_memory_bundle(self, context: ConversationContext, latest_handover_summary: dict[str, Any] | None = None) -> MemoryBundle:
        return MemoryBundle(
            recent_messages=list(context.recent_messages or []),
            recent_summary=context.recent_summary or "",
            long_term_memory=context.long_term_memory or {},
            handover_summary=latest_handover_summary or {},
        )

    def _user_state_summary(self, user_state: UserStateSnapshot) -> str:
        return (
            f"ops_category={user_state.ops_category}; project_id={user_state.project_id}; "
            f"project_segment_id={user_state.project_segment_id}; tags={user_state.tags}; "
            f"relationship_score={user_state.relationship_score}; trust_score={user_state.trust_score}; "
            f"comfort_score={user_state.comfort_score}; current_heat={user_state.current_heat}; "
            f"marketing_tolerance={user_state.marketing_tolerance}; boundary_sensitivity={user_state.boundary_sensitivity}; "
            f"response_speed_preference={user_state.response_speed_preference}; emotional_receptiveness={user_state.emotional_receptiveness}; "
            f"professional_receptiveness={user_state.professional_receptiveness}; recent_busy_score={user_state.recent_busy_score}; "
            f"push_fatigue_score={user_state.push_fatigue_score}; openness_score={user_state.openness_score}; warmth_score={user_state.warmth_score}; "
            f"project_interest_strength={user_state.current_project_interest_strength}"
        )

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

        latest_user_text = (inbound_message.text or "").strip()
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
        memory_bundle = self._build_memory_bundle(context, latest_handover_summary)
        understanding = self.understanding_engine.analyze(latest_user_text, recent_context, self.persona_core.to_summary(), self._user_state_summary(user_state))
        stage_decision = self.stage_engine.decide(understanding, context.current_stage)
        mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
        turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
        style_spec = self.humanization_controller.build_style_spec(turn_decision, understanding, stage_decision, user_state, persona_profile)
        project_decision = self.project_classifier.classify(business_account_id, understanding, context, user_state, latest_user_text)
        intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
        segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
        escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
        escalation_decision["should_queue_admin"] = bool(escalation_decision.get("should_queue_admin") or understanding.human_takeover_hint)
        if understanding.human_takeover_hint and not escalation_decision.get("reason"):
            escalation_decision["reason"] = "user asked for a real person or direct explanation"
            escalation_decision["notify_level"] = "suggest_takeover"
        tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
        ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)
        reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
        reply_plan.should_reply = turn_decision.should_reply
        reply_plan.should_continue_product = turn_decision.should_push_project
        reply_plan.should_leave_space = turn_decision.exit_strategy in ("leave_space", "soft_hold")
        reply_plan.should_self_share = turn_decision.self_disclosure_level != "none"
        selected_content = self.content_selector.select(business_account_id, project_decision.project_id, mode_decision.chat_mode, reply_plan)
        if not turn_decision.should_push_project and not understanding.explicit_product_query:
            selected_content = {k: v for k, v in (selected_content or {}).items() if k in ("persona_materials", "self_share_materials")}
        understanding_payload = understanding.__dict__.copy()
        understanding_payload["turn_decision"] = asdict(turn_decision)
        understanding_payload["style_spec"] = asdict(style_spec)
        understanding_payload["memory_summary"] = {
            "recent_messages_count": len(memory_bundle.recent_messages),
            "has_recent_summary": bool(memory_bundle.recent_summary),
            "has_handover_summary": bool(memory_bundle.handover_summary),
        }
        if latest_handover_summary:
            understanding_payload["resume_hint"] = latest_handover_summary.get("resume_suggestion")
        draft_reply = self.reply_style_engine.generate(
            latest_user_text,
            recent_context,
            persona_summary,
            self._user_state_summary(user_state),
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding_payload,
            reply_plan.__dict__,
            selected_content,
        )
        final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        final_reply = FinalReply(
            text=final_text,
            delay_seconds=delay_seconds,
            metadata={"turn_decision": asdict(turn_decision), "style_spec": asdict(style_spec)},
        )
        if reply_plan.should_reply and final_reply.text.strip():
            send_result = self.sender_service.send_text_reply(
                conversation_id,
                final_reply.text,
                final_reply.delay_seconds,
                raw_payload=inbound_message.raw_payload,
                metadata={
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "delivery": "queued",
                },
            )
            self.conversation_repo.save_message(
                conversation_id,
                "ai",
                "text",
                final_reply.text,
                {
                    "selected_content": selected_content,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "delivery": send_result,
                },
                None,
            )
            self.conversation_repo.set_last_ai_reply_at(conversation_id)
        self.conversation_repo.update_conversation_state(conversation_id, stage_decision.stage, mode_decision.chat_mode, understanding.current_mainline_should_continue)
        if project_decision.project_id is not None:
            self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
        self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_decision["ops_category"], ops_decision["reason"], "system")
        try:
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "AI Turn Decision",
                {
                    "conversation_id": conversation_id,
                    "stage": stage_decision.stage,
                    "stage_confidence": stage_decision.confidence,
                    "chat_mode": mode_decision.chat_mode,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "intent_level": intent_decision.level,
                    "intent_score": intent_decision.score,
                },
            )
        except Exception:
            logger.exception("Failed to create AI turn-decision receipt")
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
                    "style_spec": asdict(style_spec),
                },
            )
            if self.admin_notifier:
                self.admin_notifier.notify_high_intent(business_account_id, user_id, conversation_id, escalation_decision.get("notify_level") or "watch", escalation_decision.get("reason") or "")


_Prev_AsyncSenderWorker_2 = AsyncSenderWorker
class AsyncSenderWorker(_Prev_AsyncSenderWorker_2):
    def run_forever(self) -> None:
        db = Database(self.database_url)
        db.connect()
        repo = OutboundMessageJobRepository(db)
        while not self._stop_event.is_set():
            try:
                jobs = repo.fetch_ready_jobs(limit=10)
                if not jobs:
                    time.sleep(self.poll_interval)
                    continue
                for job in jobs:
                    job_id = int(job["id"])
                    payload = job.get("payload_json") or {}
                    try:
                        repo.mark_sending(job_id)
                        method = "sendMessage"
                        api_payload = {
                            "chat_id": payload.get("chat_id"),
                            "text": payload.get("text") or "",
                            "business_connection_id": payload.get("business_connection_id"),
                        }
                        if not api_payload.get("chat_id") or not api_payload.get("text"):
                            raise RuntimeError("missing delivery context for outbound reply job")
                        self._call_api(method, api_payload)
                        repo.mark_sent(job_id)
                    except Exception as exc:
                        logger.exception("Async sender worker failed job_id=%s", job_id)
                        repo.mark_failed(job_id, str(exc))
                time.sleep(0.1)
            except Exception:
                logger.exception("Async sender loop error")
                time.sleep(self.poll_interval)
        db.close()


# =========================
# V3 relationship-operating upgrade patch
# =========================

def ensure_v3_upgrade_schema(db: Database) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS followup_jobs (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            followup_type TEXT NOT NULL,
            text_payload TEXT,
            scheduled_for TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL DEFAULT 'scheduled',
            outbound_job_id BIGINT,
            note TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ]
    with db.transaction():
        with db.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)


@dataclass
class TurnDecision:
    reply_goal: str = "rapport"
    should_push_project: bool = False
    social_distance: str = "balanced"
    reply_length: str = "medium"
    ask_followup_question: bool = False
    exit_strategy: str = "gentle_close"
    self_disclosure_level: str = "light"
    reason: str = ""
    should_reply: bool = True
    marketing_intensity: str = "subtle"
    tone_bias: str = "natural"
    need_followup: bool = False
    followup_type: str | None = None
    followup_delay_seconds: int | None = None
    nurture_goal: str | None = None
    should_write_memory: bool = True


class FollowupJobRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def cancel_pending_for_conversation(self, conversation_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "UPDATE followup_jobs SET status='cancelled', updated_at=NOW() WHERE conversation_id=%s AND status='scheduled'",
                    (conversation_id,),
                )

    def insert_job(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        followup_type: str,
        text_payload: str,
        delay_seconds: int,
        note: str | None = None,
    ) -> int:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO followup_jobs (
                        business_account_id, user_id, conversation_id, followup_type,
                        text_payload, scheduled_for, status, note
                    ) VALUES (%s,%s,%s,%s,%s,NOW() + (%s * INTERVAL '1 second'),'scheduled',%s)
                    RETURNING id
                    """,
                    (business_account_id, user_id, conversation_id, followup_type, text_payload, max(int(delay_seconds or 0), 0), note),
                )
                return int(cur.fetchone()["id"])

    def attach_outbound_job(self, followup_job_id: int, outbound_job_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "UPDATE followup_jobs SET outbound_job_id=%s, updated_at=NOW() WHERE id=%s",
                    (outbound_job_id, followup_job_id),
                )


_Prev_ConversationRepository_1 = ConversationRepository
class ConversationRepository(_Prev_ConversationRepository_1):
    def save_summary(self, conversation_id: int, summary_text: str) -> None:
        if not (summary_text or "").strip():
            return
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversation_summaries (conversation_id, summary_text) VALUES (%s,%s)",
                    (conversation_id, summary_text.strip()[:3000]),
                )


class ProjectNurturePlanner:
    def plan(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        turn_decision: TurnDecision,
        project_decision: ProjectDecision,
        user_state: UserStateSnapshot,
    ) -> dict[str, Any]:
        if project_decision.project_id is None or not turn_decision.should_push_project:
            return {
                "should_include_project_content": False,
                "nurture_angle": None,
                "max_project_points": 0,
            }
        angle = "single_clear_point"
        max_points = 1
        if stage_decision.stage == "high_intent_progress":
            angle = "next_step_clarity"
            max_points = 2
        elif understanding.explicit_product_query:
            angle = "direct_answer_then_soft_bridge"
        elif stage_decision.stage in ("project_discussion", "low_intent_nurture"):
            angle = "light_value_bridge"
        if user_state.boundary_sensitivity > 0.65:
            max_points = 1
            angle = "brief_and_low_pressure"
        return {
            "should_include_project_content": True,
            "nurture_angle": angle,
            "max_project_points": max_points,
        }


class MemoryWritebackEngine:
    def __init__(self, conversation_repo: ConversationRepository) -> None:
        self.conversation_repo = conversation_repo

    def write_after_turn(
        self,
        conversation_id: int,
        latest_user_text: str,
        final_reply_text: str,
        stage_decision: StageDecision,
        turn_decision: TurnDecision,
        project_decision: ProjectDecision,
        intent_decision: IntentDecision,
    ) -> None:
        if not turn_decision.should_write_memory:
            return
        user_text = " ".join((latest_user_text or "").split())[:220]
        reply_text = " ".join((final_reply_text or "").split())[:220]
        summary = (
            f"stage={stage_decision.stage}; goal={turn_decision.reply_goal}; "
            f"project_id={project_decision.project_id}; intent={intent_decision.level}; "
            f"user='{user_text}'; ai='{reply_text}'"
        )
        self.conversation_repo.save_summary(conversation_id, summary)


class FollowupScheduler:
    def __init__(self, followup_repo: FollowupJobRepository, outbound_queue: OutboundMessageQueue, receipt_repo: ReceiptRepository) -> None:
        self.followup_repo = followup_repo
        self.outbound_queue = outbound_queue
        self.receipt_repo = receipt_repo

    def schedule_after_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        turn_decision: TurnDecision,
        user_state: UserStateSnapshot,
    ) -> dict[str, Any] | None:
        self.followup_repo.cancel_pending_for_conversation(conversation_id)
        if understanding.boundary_signal or user_state.boundary_sensitivity > 0.78:
            return None
        if not turn_decision.need_followup:
            return None
        followup_type = turn_decision.followup_type or "light_checkin"
        delay_seconds = int(turn_decision.followup_delay_seconds or 0)
        if delay_seconds <= 0:
            return None
        text = self._build_followup_text(followup_type, stage_decision, user_state)
        if not text:
            return None
        followup_job_id = self.followup_repo.insert_job(
            business_account_id=business_account_id,
            user_id=user_id,
            conversation_id=conversation_id,
            followup_type=followup_type,
            text_payload=text,
            delay_seconds=delay_seconds,
            note=turn_decision.reason,
        )
        outbound_job_id = self.outbound_queue.enqueue_text_reply(
            conversation_id=conversation_id,
            text=text,
            raw_payload=None,
            delay_seconds=delay_seconds,
            metadata={"followup_type": followup_type, "delivery": "queued_followup"},
        )
        self.followup_repo.attach_outbound_job(followup_job_id, outbound_job_id)
        try:
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "Follow-up scheduled",
                {
                    "conversation_id": conversation_id,
                    "followup_type": followup_type,
                    "delay_seconds": delay_seconds,
                    "followup_job_id": followup_job_id,
                    "outbound_job_id": outbound_job_id,
                },
            )
        except Exception:
            logger.exception("Failed to create follow-up receipt")
        return {
            "followup_type": followup_type,
            "delay_seconds": delay_seconds,
            "followup_job_id": followup_job_id,
            "outbound_job_id": outbound_job_id,
        }

    def _build_followup_text(self, followup_type: str, stage_decision: StageDecision, user_state: UserStateSnapshot) -> str:
        if followup_type == "high_intent_reconnect":
            return "Just checking in lightly — whenever you have a moment, I can walk you through the next part step by step."
        if followup_type == "rapport_keep_warm":
            return "Just dropping a light hello — hope your day’s been going gently. No pressure to reply, I just wanted to say hi."
        if followup_type == "resume_after_handover":
            return "Just picking this up gently from where we left it — whenever you’re free, I’m here."
        if stage_decision.stage in ("trust_building", "daily_rapport") or user_state.warmth_score >= 0.62:
            return "A light little check-in from me — hope your day’s been kind."
        return "Just checking in lightly — I’m here whenever you feel like continuing."


_Prev_ConversationStageEngine_3 = ConversationStageEngine
class ConversationStageEngine(_Prev_ConversationStageEngine_3):
    def decide(self, understanding: UnderstandingResult, current_stage: str | None) -> StageDecision:
        target = current_stage or "first_contact"
        reason = "retain current stage"
        confidence = 0.66
        if understanding.boundary_signal:
            target = "boundary_protection"
            reason = f"boundary detected: {understanding.boundary_signal}"
            confidence = max(0.84, understanding.boundary_strength)
        elif current_stage == "boundary_protection" and not understanding.boundary_signal and understanding.social_openness >= 0.42:
            target = "silence_recovery"
            reason = "boundary eased and user reopened"
            confidence = 0.72
        elif understanding.high_intent_signal:
            target = "high_intent_progress"
            reason = "high intent detected"
            confidence = 0.9
        elif understanding.explicit_product_query or understanding.project_relevance >= 0.77:
            target = "project_discussion"
            reason = "project discussion detected"
            confidence = 0.8
        elif understanding.project_relevance >= 0.48 and current_stage in ("trust_building", "daily_rapport", "project_discussion"):
            target = "low_intent_nurture"
            reason = "project signal present but not strong enough for active progression"
            confidence = 0.68
        elif current_stage in (None, "", "new", "relationship", "first_contact"):
            if understanding.social_openness >= 0.6 or understanding.emotion_state in ("positive", "low"):
                target = "trust_building"
                reason = "rapport or emotional openness detected"
                confidence = 0.72
            else:
                target = "daily_rapport"
                reason = "light rapport maintenance"
                confidence = 0.64
        return StageDecision(stage=target, changed=(current_stage != target), reason=reason, confidence=confidence)


_Prev_TurnDecisionEngine_3 = TurnDecisionEngine
class TurnDecisionEngine(_Prev_TurnDecisionEngine_3):
    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        if understanding.boundary_signal:
            return TurnDecision(
                reply_goal="respect_boundary",
                should_push_project=False,
                social_distance="give_space",
                reply_length="short",
                ask_followup_question=False,
                exit_strategy="leave_space",
                self_disclosure_level="none",
                reason=f"boundary:{understanding.boundary_signal}",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="gentle",
                need_followup=False,
                followup_type=None,
                nurture_goal=None,
            )
        if understanding.high_intent_signal:
            ask_followup = user_state.boundary_sensitivity < 0.65 and user_state.push_fatigue_score < 0.7
            return TurnDecision(
                reply_goal="progress_without_pressure",
                should_push_project=True,
                social_distance="professional_close",
                reply_length="medium",
                ask_followup_question=ask_followup,
                exit_strategy="soft_next_step",
                self_disclosure_level="none",
                reason="high_intent_signal",
                should_reply=True,
                marketing_intensity="light",
                tone_bias="professional_warm",
                need_followup=True,
                followup_type="high_intent_reconnect",
                followup_delay_seconds=6 * 3600,
                nurture_goal="move_one_step_forward",
            )
        if stage_decision.stage == "low_intent_nurture":
            return TurnDecision(
                reply_goal="nurture_interest",
                should_push_project=True,
                social_distance="balanced",
                reply_length="medium",
                ask_followup_question=user_state.boundary_sensitivity < 0.5,
                exit_strategy="light_hook",
                self_disclosure_level="none",
                reason="low_intent_nurture",
                should_reply=True,
                marketing_intensity="subtle",
                tone_bias="natural",
                need_followup=True,
                followup_type="rapport_keep_warm",
                followup_delay_seconds=20 * 3600,
                nurture_goal="keep_interest_alive",
            )
        if stage_decision.stage == "silence_recovery":
            return TurnDecision(
                reply_goal="reopen_gently",
                should_push_project=False,
                social_distance="warm_supportive",
                reply_length="short",
                ask_followup_question=False,
                exit_strategy="soft_hold",
                self_disclosure_level="none",
                reason="silence_recovery",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="soft",
                need_followup=False,
                nurture_goal="rebuild_safety",
            )
        if understanding.explicit_product_query or understanding.project_relevance >= 0.72:
            return TurnDecision(
                reply_goal="answer_and_transition",
                should_push_project=True,
                social_distance="balanced",
                reply_length="medium",
                ask_followup_question=user_state.boundary_sensitivity < 0.55,
                exit_strategy="gentle_close",
                self_disclosure_level="none",
                reason="product_focus",
                should_reply=True,
                marketing_intensity="subtle",
                tone_bias="clear",
                need_followup=True if user_state.current_project_interest_strength >= 0.5 else False,
                followup_type="high_intent_reconnect" if user_state.current_project_interest_strength >= 0.7 else None,
                followup_delay_seconds=8 * 3600 if user_state.current_project_interest_strength >= 0.7 else None,
                nurture_goal="clarify_one_point",
            )
        if understanding.emotion_state == "low":
            return TurnDecision(
                reply_goal="support_first",
                should_push_project=False,
                social_distance="warm_supportive",
                reply_length="short",
                ask_followup_question=False,
                exit_strategy="soft_hold",
                self_disclosure_level="none",
                reason="emotion_low",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="soft",
                need_followup=False,
                nurture_goal="stabilize_emotion",
            )
        if stage_decision.stage in ("trust_building", "daily_rapport"):
            return TurnDecision(
                reply_goal="build_rapport",
                should_push_project=False,
                social_distance="friendly_but_measured",
                reply_length="short" if user_state.response_speed_preference == "slow" else "medium",
                ask_followup_question=user_state.boundary_sensitivity < 0.45,
                exit_strategy="light_hook" if user_state.boundary_sensitivity < 0.45 else "gentle_close",
                self_disclosure_level="light" if understanding.social_openness >= 0.55 else "none",
                reason="rapport_stage",
                should_reply=True,
                marketing_intensity="none",
                tone_bias="natural",
                need_followup=user_state.openness_score >= 0.6 and user_state.boundary_sensitivity < 0.45,
                followup_type="rapport_keep_warm" if user_state.openness_score >= 0.6 and user_state.boundary_sensitivity < 0.45 else None,
                followup_delay_seconds=24 * 3600 if user_state.openness_score >= 0.6 and user_state.boundary_sensitivity < 0.45 else None,
                nurture_goal="strengthen_connection",
            )
        return TurnDecision(
            reply_goal="maintain_presence",
            should_push_project=False,
            social_distance="balanced",
            reply_length="medium",
            ask_followup_question=False,
            exit_strategy="gentle_close",
            self_disclosure_level="none",
            reason="default_v3",
            should_reply=True,
            marketing_intensity="none" if user_state.boundary_sensitivity > 0.55 else "subtle",
            tone_bias="natural",
            need_followup=False,
            nurture_goal="stay_present",
        )


_Prev_HumanizationController_2 = HumanizationController
class HumanizationController(_Prev_HumanizationController_2):
    def build_style_spec(
        self,
        decision: TurnDecision,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        persona_profile: dict[str, Any] | None = None,
    ) -> StyleSpec:
        spec = super().build_style_spec(decision, understanding, stage_decision, user_state, persona_profile)
        if stage_decision.stage in ("trust_building", "daily_rapport", "silence_recovery"):
            spec.warmth = "medium_high"
            if spec.formality == "balanced":
                spec.formality = "casual"
        if stage_decision.stage in ("project_discussion", "high_intent_progress", "low_intent_nurture") and decision.should_push_project:
            spec.marketing_visibility = "subtle" if decision.marketing_intensity in ("none", "subtle") else decision.marketing_intensity
            spec.completion_level = "focused"
        if decision.exit_strategy in ("soft_hold", "leave_space"):
            spec.question_rate = "none"
        return spec


_Prev_Orchestrator_4 = Orchestrator
class Orchestrator(_Prev_Orchestrator_4):
    def __init__(
        self,
        *args,
        turn_decision_engine: TurnDecisionEngine | None = None,
        humanization_controller: HumanizationController | None = None,
        project_nurture_planner: ProjectNurturePlanner | None = None,
        memory_writeback_engine: MemoryWritebackEngine | None = None,
        followup_scheduler: FollowupScheduler | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, turn_decision_engine=turn_decision_engine, humanization_controller=humanization_controller, **kwargs)
        self.project_nurture_planner = project_nurture_planner or ProjectNurturePlanner()
        self.memory_writeback_engine = memory_writeback_engine
        self.followup_scheduler = followup_scheduler

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

        latest_user_text = (inbound_message.text or "").strip()
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
        memory_bundle = self._build_memory_bundle(context, latest_handover_summary)
        understanding = self.understanding_engine.analyze(latest_user_text, recent_context, self.persona_core.to_summary(), self._user_state_summary(user_state))
        stage_decision = self.stage_engine.decide(understanding, context.current_stage)
        mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
        turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
        style_spec = self.humanization_controller.build_style_spec(turn_decision, understanding, stage_decision, user_state, persona_profile)
        project_decision = self.project_classifier.classify(business_account_id, understanding, context, user_state, latest_user_text)
        nurture_plan = self.project_nurture_planner.plan(understanding, stage_decision, turn_decision, project_decision, user_state)
        intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
        segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
        escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
        escalation_decision["should_queue_admin"] = bool(escalation_decision.get("should_queue_admin") or understanding.human_takeover_hint)
        if understanding.human_takeover_hint and not escalation_decision.get("reason"):
            escalation_decision["reason"] = "user asked for a real person or direct explanation"
            escalation_decision["notify_level"] = "suggest_takeover"
        tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
        ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)
        reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
        reply_plan.should_reply = turn_decision.should_reply
        reply_plan.should_continue_product = turn_decision.should_push_project
        reply_plan.should_leave_space = turn_decision.exit_strategy in ("leave_space", "soft_hold")
        reply_plan.should_self_share = turn_decision.self_disclosure_level != "none"
        selected_content = self.content_selector.select(business_account_id, project_decision.project_id, mode_decision.chat_mode, reply_plan)
        if not turn_decision.should_push_project and not understanding.explicit_product_query:
            selected_content = {k: v for k, v in (selected_content or {}).items() if k in ("persona_materials", "self_share_materials")}
        understanding_payload = understanding.__dict__.copy()
        understanding_payload["turn_decision"] = asdict(turn_decision)
        understanding_payload["style_spec"] = asdict(style_spec)
        understanding_payload["nurture_plan"] = nurture_plan
        understanding_payload["memory_summary"] = {
            "recent_messages_count": len(memory_bundle.recent_messages),
            "has_recent_summary": bool(memory_bundle.recent_summary),
            "has_handover_summary": bool(memory_bundle.handover_summary),
        }
        if latest_handover_summary:
            understanding_payload["resume_hint"] = latest_handover_summary.get("resume_suggestion")
        draft_reply = self.reply_style_engine.generate(
            latest_user_text,
            recent_context,
            persona_summary,
            self._user_state_summary(user_state),
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding_payload,
            reply_plan.__dict__,
            selected_content,
        )
        final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        final_reply = FinalReply(
            text=final_text,
            delay_seconds=delay_seconds,
            metadata={
                "turn_decision": asdict(turn_decision),
                "style_spec": asdict(style_spec),
                "nurture_plan": nurture_plan,
            },
        )
        send_result = None
        if reply_plan.should_reply and final_reply.text.strip():
            send_result = self.sender_service.send_text_reply(
                conversation_id,
                final_reply.text,
                final_reply.delay_seconds,
                raw_payload=inbound_message.raw_payload,
                metadata={
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "nurture_plan": nurture_plan,
                    "delivery": "queued",
                },
            )
            self.conversation_repo.save_message(
                conversation_id,
                "ai",
                "text",
                final_reply.text,
                {
                    "selected_content": selected_content,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "nurture_plan": nurture_plan,
                    "delivery": send_result,
                },
                None,
            )
            self.conversation_repo.set_last_ai_reply_at(conversation_id)
        self.conversation_repo.update_conversation_state(conversation_id, stage_decision.stage, mode_decision.chat_mode, understanding.current_mainline_should_continue)
        if project_decision.project_id is not None:
            self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
        self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_decision["ops_category"], ops_decision["reason"], "system")
        if self.memory_writeback_engine and final_reply.text.strip():
            try:
                self.memory_writeback_engine.write_after_turn(
                    conversation_id,
                    latest_user_text,
                    final_reply.text,
                    stage_decision,
                    turn_decision,
                    project_decision,
                    intent_decision,
                )
            except Exception:
                logger.exception("Failed to write conversation memory summary")
        followup_result = None
        if self.followup_scheduler and final_reply.text.strip():
            try:
                followup_result = self.followup_scheduler.schedule_after_turn(
                    business_account_id,
                    user_id,
                    conversation_id,
                    understanding,
                    stage_decision,
                    turn_decision,
                    user_state,
                )
            except Exception:
                logger.exception("Failed to schedule follow-up")
        try:
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "AI Turn Decision",
                {
                    "conversation_id": conversation_id,
                    "stage": stage_decision.stage,
                    "stage_confidence": stage_decision.confidence,
                    "chat_mode": mode_decision.chat_mode,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "intent_level": intent_decision.level,
                    "intent_score": intent_decision.score,
                    "nurture_plan": nurture_plan,
                    "followup_result": followup_result,
                },
            )
        except Exception:
            logger.exception("Failed to create AI turn-decision receipt")
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
                    "style_spec": asdict(style_spec),
                },
            )
            if self.admin_notifier:
                self.admin_notifier.notify_high_intent(business_account_id, user_id, conversation_id, escalation_decision.get("notify_level") or "watch", escalation_decision.get("reason") or "")


def build_app_components(settings: Settings) -> dict[str, Any]:
    db = Database(settings.database_url)
    db.connect()
    initialize_database(db)
    ensure_v1_upgrade_schema(db)
    ensure_v3_upgrade_schema(db)

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
    processed_update_repo = ProcessedUpdateRepository(db)
    outbound_message_job_repo = OutboundMessageJobRepository(db)
    followup_repo = FollowupJobRepository(db)

    openai_adapter = OpenAIClientAdapter(settings.openai_api_key)
    llm_service = LLMService(openai_adapter, settings.llm_model_name)
    outbound_queue = OutboundMessageQueue(outbound_message_job_repo, tg_client)
    sender_service = SenderService(TelegramBusinessSenderAdapter(tg_client), outbound_queue=outbound_queue)
    admin_notifier = AdminNotifier(tg_client, admin_chat_ids=settings.admin_chat_ids)
    idempotency_guard = WebhookIdempotencyGuard(processed_update_repo)
    sender_worker = AsyncSenderWorker(settings.database_url, settings.tg_bot_token)

    persona_core = PersonaCore()
    persona_profile_builder = PersonaProfileBuilder(material_repo)
    understanding_engine = UserUnderstandingEngine(llm_service)
    stage_engine = ConversationStageEngine()
    mode_router = ChatModeRouter()
    reply_planner = ReplyPlanner()
    reply_style_engine = ReplyStyleEngine(llm_service)
    reply_self_check_engine = ReplySelfCheckEngine()
    reply_delay_engine = ReplyDelayEngine()
    turn_decision_engine = TurnDecisionEngine()
    humanization_controller = HumanizationController()
    project_nurture_planner = ProjectNurturePlanner()
    memory_writeback_engine = MemoryWritebackEngine(conversation_repo)
    followup_scheduler = FollowupScheduler(followup_repo, outbound_queue, receipt_repo)

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
        turn_decision_engine=turn_decision_engine,
        humanization_controller=humanization_controller,
        project_nurture_planner=project_nurture_planner,
        memory_writeback_engine=memory_writeback_engine,
        followup_scheduler=followup_scheduler,
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
        "idempotency_guard": idempotency_guard,
        "sender_worker": sender_worker,
        "followup_repo": followup_repo,
    }

def main() -> None:
    settings = Settings.load()
    settings.validate()
    setup_logging(settings.log_level)
    logger.info("Starting application.")
    app_components = build_app_components(settings)
    sender_worker: AsyncSenderWorker = app_components["sender_worker"]
    sender_worker.start()
    logger.info("Async sender worker started.")

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



# =========================
# V4 complete-operations upgrade patch
# =========================

def ensure_v4_upgrade_schema(db: Database) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS decision_audit_logs (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            conversation_id BIGINT REFERENCES conversations(id) ON DELETE SET NULL,
            decision_type TEXT NOT NULL,
            decision_result TEXT,
            confidence DOUBLE PRECISION,
            reason_text TEXT,
            evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            source TEXT NOT NULL DEFAULT 'system',
            operator TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS override_lock_records (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            entity_type TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            action_type TEXT NOT NULL,
            operator TEXT,
            reason_text TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ]
    with db.transaction():
        with db.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_decision_audit_logs_biz_user ON decision_audit_logs(business_account_id, user_id, created_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_override_lock_records_biz_user ON override_lock_records(business_account_id, user_id, created_at DESC)")


_Prev_ReceiptRepository_1 = ReceiptRepository
class ReceiptRepository(_Prev_ReceiptRepository_1):
    def create_admin_receipt(self, business_account_id: int, user_id: int, receipt_type: str, title: str, content_json: dict, status: str = "pending") -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO receipts (
                        business_account_id, user_id, receipt_type, title, content_text, content_json, status, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
                    """,
                    (business_account_id, user_id, receipt_type, title, title, Json(content_json or {}), status),
                )

    def list_pending_receipts(self, business_account_id: int, limit: int = 50) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM receipts WHERE business_account_id=%s AND status='pending' ORDER BY created_at DESC LIMIT %s",
                (business_account_id, limit),
            )
            return cur.fetchall()


class DecisionAuditRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def log(
        self,
        business_account_id: int,
        user_id: int | None,
        conversation_id: int | None,
        decision_type: str,
        decision_result: str | None,
        reason_text: str,
        evidence_json: dict | None = None,
        confidence: float | None = None,
        source: str = "system",
        operator: str | None = None,
    ) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO decision_audit_logs (
                        business_account_id, user_id, conversation_id, decision_type,
                        decision_result, confidence, reason_text, evidence_json, source, operator
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        business_account_id,
                        user_id,
                        conversation_id,
                        decision_type,
                        decision_result,
                        confidence,
                        reason_text,
                        Json(evidence_json or {}),
                        source,
                        operator,
                    ),
                )

    def list_recent_for_user(self, business_account_id: int, user_id: int, limit: int = 50) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM decision_audit_logs WHERE business_account_id=%s AND user_id=%s ORDER BY created_at DESC LIMIT %s",
                (business_account_id, user_id, limit),
            )
            return cur.fetchall()

    def count_for_business(self, business_account_id: int, decision_type: str | None = None) -> int:
        with self.db.cursor() as cur:
            if decision_type:
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM decision_audit_logs WHERE business_account_id=%s AND decision_type=%s",
                    (business_account_id, decision_type),
                )
            else:
                cur.execute("SELECT COUNT(*) AS cnt FROM decision_audit_logs WHERE business_account_id=%s", (business_account_id,))
            row = cur.fetchone()
            return int((row or {}).get("cnt") or 0)


class OverrideLockRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def log_record(
        self,
        business_account_id: int,
        user_id: int,
        entity_type: str,
        entity_key: str,
        old_value: str | None,
        new_value: str | None,
        action_type: str,
        operator: str,
        reason_text: str,
    ) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO override_lock_records (
                        business_account_id, user_id, entity_type, entity_key,
                        old_value, new_value, action_type, operator, reason_text
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (business_account_id, user_id, entity_type, entity_key, old_value, new_value, action_type, operator, reason_text),
                )

    def list_recent_for_user(self, business_account_id: int, user_id: int, limit: int = 50) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM override_lock_records WHERE business_account_id=%s AND user_id=%s ORDER BY created_at DESC LIMIT %s",
                (business_account_id, user_id, limit),
            )
            return cur.fetchall()

    def get_lock_summary(self, business_account_id: int, user_id: int) -> dict[str, Any]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT project_id, is_locked FROM user_project_state WHERE business_account_id=%s AND user_id=%s LIMIT 1",
                (business_account_id, user_id),
            )
            project_row = cur.fetchone() or {}
            cur.execute(
                """
                SELECT t.name FROM user_tags ut
                JOIN tags t ON ut.tag_id=t.id
                WHERE ut.business_account_id=%s AND ut.user_id=%s AND ut.is_active=TRUE AND ut.is_locked=TRUE
                ORDER BY t.name ASC
                """,
                (business_account_id, user_id),
            )
            tag_rows = cur.fetchall()
        return {
            "project_locked": bool(project_row.get("is_locked")),
            "locked_project_id": project_row.get("project_id"),
            "locked_tags": [r["name"] for r in tag_rows],
        }

    def lock_project(self, business_account_id: int, user_id: int, operator: str, reason_text: str) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "SELECT project_id, is_locked FROM user_project_state WHERE business_account_id=%s AND user_id=%s LIMIT 1",
                    (business_account_id, user_id),
                )
                row = cur.fetchone()
                if not row:
                    return {"ok": False, "reason": "project state not found"}
                cur.execute(
                    "UPDATE user_project_state SET is_locked=TRUE, updated_by=%s, updated_at=NOW() WHERE business_account_id=%s AND user_id=%s",
                    (operator, business_account_id, user_id),
                )
                self.log_record(business_account_id, user_id, "project", "project_state", str(row.get("project_id")), str(row.get("project_id")), "lock", operator, reason_text)
                return {"ok": True, "project_id": row.get("project_id")}

    def unlock_project(self, business_account_id: int, user_id: int, operator: str, reason_text: str) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "SELECT project_id, is_locked FROM user_project_state WHERE business_account_id=%s AND user_id=%s LIMIT 1",
                    (business_account_id, user_id),
                )
                row = cur.fetchone()
                if not row:
                    return {"ok": False, "reason": "project state not found"}
                cur.execute(
                    "UPDATE user_project_state SET is_locked=FALSE, updated_by=%s, updated_at=NOW() WHERE business_account_id=%s AND user_id=%s",
                    (operator, business_account_id, user_id),
                )
                self.log_record(business_account_id, user_id, "project", "project_state", str(row.get("project_id")), str(row.get("project_id")), "unlock", operator, reason_text)
                return {"ok": True, "project_id": row.get("project_id")}

    def lock_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("SELECT id FROM tags WHERE business_account_id=%s AND name=%s LIMIT 1", (business_account_id, tag_name))
                tag_row = cur.fetchone()
                if not tag_row:
                    return {"ok": False, "reason": f"tag not found: {tag_name}"}
                cur.execute(
                    "UPDATE user_tags SET is_locked=TRUE, updated_at=NOW() WHERE business_account_id=%s AND user_id=%s AND tag_id=%s AND is_active=TRUE",
                    (business_account_id, user_id, tag_row["id"]),
                )
                self.log_record(business_account_id, user_id, "tag", tag_name, tag_name, tag_name, "lock", operator, reason_text)
                return {"ok": True, "tag_name": tag_name}

    def unlock_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("SELECT id FROM tags WHERE business_account_id=%s AND name=%s LIMIT 1", (business_account_id, tag_name))
                tag_row = cur.fetchone()
                if not tag_row:
                    return {"ok": False, "reason": f"tag not found: {tag_name}"}
                cur.execute(
                    "UPDATE user_tags SET is_locked=FALSE, updated_at=NOW() WHERE business_account_id=%s AND user_id=%s AND tag_id=%s",
                    (business_account_id, user_id, tag_row["id"]),
                )
                self.log_record(business_account_id, user_id, "tag", tag_name, tag_name, tag_name, "unlock", operator, reason_text)
                return {"ok": True, "tag_name": tag_name}


_Prev_UserRepository_3 = UserRepository
class UserRepository(_Prev_UserRepository_3):
    def __init__(self, db: Database, decision_audit_repo: DecisionAuditRepository | None = None) -> None:
        super().__init__(db)
        self.decision_audit_repo = decision_audit_repo

    def update_project_state(self, business_account_id: int, user_id: int, project_id: int | None, reason: str) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT project_id, is_locked FROM user_project_state WHERE business_account_id=%s AND user_id=%s LIMIT 1",
                (business_account_id, user_id),
            )
            row = cur.fetchone() or {}
        if row.get("is_locked"):
            if self.decision_audit_repo:
                self.decision_audit_repo.log(
                    business_account_id,
                    user_id,
                    None,
                    "project_update_skipped_locked",
                    str(row.get("project_id")),
                    "AI project update skipped because project state is locked",
                    {"attempted_project_id": project_id, "locked_project_id": row.get("project_id"), "reason": reason},
                    source="system",
                )
            return
        super().update_project_state(business_account_id, user_id, project_id, reason)

    def apply_tag_decision(self, business_account_id: int, user_id: int, tag_decision: TagDecision) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT t.name FROM user_tags ut JOIN tags t ON ut.tag_id=t.id
                WHERE ut.business_account_id=%s AND ut.user_id=%s AND ut.is_locked=TRUE
                """,
                (business_account_id, user_id),
            )
            locked_rows = cur.fetchall()
        locked_tags = {r["name"] for r in (locked_rows or [])}
        filtered = TagDecision(
            add_tags=[t for t in tag_decision.add_tags if t not in locked_tags],
            remove_tags=[t for t in tag_decision.remove_tags if t not in locked_tags],
            reason=tag_decision.reason,
        )
        if locked_tags and self.decision_audit_repo:
            self.decision_audit_repo.log(
                business_account_id,
                user_id,
                None,
                "tag_update_skipped_locked",
                ",".join(sorted(locked_tags)),
                "AI tag update skipped for locked tags",
                {"locked_tags": sorted(locked_tags), "attempted_add": tag_decision.add_tags, "attempted_remove": tag_decision.remove_tags},
                source="system",
            )
        super().apply_tag_decision(business_account_id, user_id, filtered)


class ReceiptCenter:
    def __init__(self, receipt_repo: ReceiptRepository, decision_audit_repo: DecisionAuditRepository) -> None:
        self.receipt_repo = receipt_repo
        self.decision_audit_repo = decision_audit_repo

    def log_decision(self, business_account_id: int, user_id: int, conversation_id: int | None, decision_type: str, result: str, reason_text: str, evidence: dict | None = None, confidence: float | None = None, source: str = "system", operator: str | None = None, create_receipt: bool = False, title: str | None = None) -> None:
        self.decision_audit_repo.log(
            business_account_id,
            user_id,
            conversation_id,
            decision_type,
            result,
            reason_text,
            evidence,
            confidence,
            source,
            operator,
        )
        if create_receipt:
            self.receipt_repo.create_admin_receipt(
                business_account_id,
                user_id,
                decision_type,
                title or decision_type,
                {
                    "conversation_id": conversation_id,
                    "decision_type": decision_type,
                    "result": result,
                    "reason": reason_text,
                    "evidence": evidence or {},
                    "source": source,
                    "operator": operator,
                },
                status="pending",
            )


_Prev_CustomerActions_1 = CustomerActions
class CustomerActions(_Prev_CustomerActions_1):
    def __init__(self, *args, receipt_center: ReceiptCenter | None = None, override_lock_repo: OverrideLockRepository | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.receipt_center = receipt_center
        self.override_lock_repo = override_lock_repo

    def start_handover(self, business_account_id: int, user_id: int, conversation_id: int, operator: str, reason: str = "manual handover") -> dict:
        result = super().start_handover(business_account_id, user_id, conversation_id, operator, reason)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(
                business_account_id, user_id, conversation_id,
                "manual_handover_started", "active", reason,
                {"already_active": bool(result.get("already_active"))},
                source="manual", operator=operator, create_receipt=True, title="Manual handover started"
            )
        return result

    def end_handover(self, conversation_id: int, session_id: int, operator: str) -> dict:
        with self.conversation_repo.db.cursor() as cur:
            cur.execute("SELECT business_account_id, user_id FROM conversations WHERE id=%s LIMIT 1", (conversation_id,))
            row = cur.fetchone() or {}
        result = super().end_handover(conversation_id, session_id, operator)
        if self.receipt_center and result.get("ok") and row:
            self.receipt_center.log_decision(
                int(row["business_account_id"]), int(row["user_id"]), conversation_id,
                "manual_handover_ended", "pending_resume", result.get("reason") or "handover ended",
                {"session_id": session_id, "summary": result.get("summary")},
                source="manual", operator=operator, create_receipt=True, title="Manual handover ended"
            )
        return result

    def enable_ai(self, conversation_id: int, operator: str) -> dict:
        with self.conversation_repo.db.cursor() as cur:
            cur.execute("SELECT business_account_id, user_id FROM conversations WHERE id=%s LIMIT 1", (conversation_id,))
            row = cur.fetchone() or {}
        result = super().enable_ai(conversation_id, operator)
        if self.receipt_center and result.get("ok") and row:
            self.receipt_center.log_decision(int(row["business_account_id"]), int(row["user_id"]), conversation_id, "ai_enabled", "enabled", result.get("reason") or "AI enabled", {"mode": "manual_action"}, source="manual", operator=operator)
        return result

    def disable_ai(self, conversation_id: int, operator: str) -> dict:
        with self.conversation_repo.db.cursor() as cur:
            cur.execute("SELECT business_account_id, user_id FROM conversations WHERE id=%s LIMIT 1", (conversation_id,))
            row = cur.fetchone() or {}
        result = super().disable_ai(conversation_id, operator)
        if self.receipt_center and result.get("ok") and row:
            self.receipt_center.log_decision(int(row["business_account_id"]), int(row["user_id"]), conversation_id, "ai_disabled", "disabled", result.get("reason") or "AI disabled", {"mode": "manual_action"}, source="manual", operator=operator)
        return result

    def resume_ai(self, conversation_id: int, operator: str) -> dict:
        with self.conversation_repo.db.cursor() as cur:
            cur.execute("SELECT business_account_id, user_id FROM conversations WHERE id=%s LIMIT 1", (conversation_id,))
            row = cur.fetchone() or {}
        result = super().resume_ai(conversation_id, operator)
        if self.receipt_center and result.get("ok") and row:
            self.receipt_center.log_decision(int(row["business_account_id"]), int(row["user_id"]), conversation_id, "ai_resumed", result.get("mode") or "resumed", result.get("reason") or "AI resumed", {"opening_hint": result.get("opening_hint")}, source="manual", operator=operator, create_receipt=True, title="AI resumed")
        return result

    def set_project(self, business_account_id: int, user_id: int, project_id: int | None, operator: str, reason_text: str = "manual project update") -> dict:
        result = super().set_project(business_account_id, user_id, project_id, operator, reason_text)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(business_account_id, user_id, None, "manual_project_set", str(project_id), reason_text, {"project_id": project_id}, source="manual", operator=operator, create_receipt=True, title="Manual project updated")
        return result

    def add_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str = "manual tag add") -> dict:
        result = super().add_tag(business_account_id, user_id, tag_name, operator, reason_text)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(business_account_id, user_id, None, "manual_tag_add", tag_name, reason_text, {"tag_name": tag_name}, source="manual", operator=operator)
        return result

    def remove_tag(self, business_account_id: int, user_id: int, tag_name: str) -> dict:
        result = super().remove_tag(business_account_id, user_id, tag_name)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(business_account_id, user_id, None, "manual_tag_remove", tag_name, "manual tag removed", {"tag_name": tag_name}, source="manual")
        return result

    def set_ops_category(self, business_account_id: int, user_id: int, ops_category: str, operator: str, reason_text: str = "manual ops category update") -> dict:
        result = super().set_ops_category(business_account_id, user_id, ops_category, operator, reason_text)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(business_account_id, user_id, None, "manual_ops_category_set", ops_category, reason_text, {"ops_category": ops_category}, source="manual", operator=operator)
        return result

    def lock_project(self, business_account_id: int, user_id: int, operator: str, reason_text: str = "manual project lock") -> dict:
        if not self.override_lock_repo:
            return {"ok": False, "reason": "override lock repo not configured"}
        result = self.override_lock_repo.lock_project(business_account_id, user_id, operator, reason_text)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(business_account_id, user_id, None, "project_lock", str(result.get("project_id")), reason_text, {"project_id": result.get("project_id")}, source="manual", operator=operator, create_receipt=True, title="Project locked")
        return result

    def unlock_project(self, business_account_id: int, user_id: int, operator: str, reason_text: str = "manual project unlock") -> dict:
        if not self.override_lock_repo:
            return {"ok": False, "reason": "override lock repo not configured"}
        result = self.override_lock_repo.unlock_project(business_account_id, user_id, operator, reason_text)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(business_account_id, user_id, None, "project_unlock", str(result.get("project_id")), reason_text, {"project_id": result.get("project_id")}, source="manual", operator=operator)
        return result

    def lock_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str = "manual tag lock") -> dict:
        if not self.override_lock_repo:
            return {"ok": False, "reason": "override lock repo not configured"}
        result = self.override_lock_repo.lock_tag(business_account_id, user_id, tag_name, operator, reason_text)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(business_account_id, user_id, None, "tag_lock", tag_name, reason_text, {"tag_name": tag_name}, source="manual", operator=operator)
        return result

    def unlock_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str = "manual tag unlock") -> dict:
        if not self.override_lock_repo:
            return {"ok": False, "reason": "override lock repo not configured"}
        result = self.override_lock_repo.unlock_tag(business_account_id, user_id, tag_name, operator, reason_text)
        if self.receipt_center and result.get("ok"):
            self.receipt_center.log_decision(business_account_id, user_id, None, "tag_unlock", tag_name, reason_text, {"tag_name": tag_name}, source="manual", operator=operator)
        return result


_Prev_AdminAPIService_1 = AdminAPIService
class AdminAPIService(_Prev_AdminAPIService_1):
    def __init__(self, *args, decision_audit_repo: DecisionAuditRepository | None = None, override_lock_repo: OverrideLockRepository | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decision_audit_repo = decision_audit_repo
        self.override_lock_repo = override_lock_repo

    def get_customer_detail(self, business_account_id: int, user_id: int) -> dict:
        data = super().get_customer_detail(business_account_id, user_id)
        if not data.get("ok"):
            return data
        if self.decision_audit_repo:
            data["decision_logs"] = self.decision_audit_repo.list_recent_for_user(business_account_id, user_id, 30)
        if self.override_lock_repo:
            data["override_records"] = self.override_lock_repo.list_recent_for_user(business_account_id, user_id, 30)
            data["lock_summary"] = self.override_lock_repo.get_lock_summary(business_account_id, user_id)
        return data

    def list_decision_logs(self, business_account_id: int, user_id: int, limit: int = 50) -> list[dict]:
        return [] if not self.decision_audit_repo else self.decision_audit_repo.list_recent_for_user(business_account_id, user_id, limit)

    def list_pending_review_receipts(self, business_account_id: int, limit: int = 50) -> list[dict]:
        return self.receipt_repo.list_pending_receipts(business_account_id, limit)

    def lock_project(self, business_account_id: int, user_id: int, operator: str, reason_text: str = "manual project lock") -> dict:
        return self.customer_actions.lock_project(business_account_id, user_id, operator, reason_text)

    def unlock_project(self, business_account_id: int, user_id: int, operator: str, reason_text: str = "manual project unlock") -> dict:
        return self.customer_actions.unlock_project(business_account_id, user_id, operator, reason_text)

    def lock_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str = "manual tag lock") -> dict:
        return self.customer_actions.lock_tag(business_account_id, user_id, tag_name, operator, reason_text)

    def unlock_tag(self, business_account_id: int, user_id: int, tag_name: str, operator: str, reason_text: str = "manual tag unlock") -> dict:
        return self.customer_actions.unlock_tag(business_account_id, user_id, tag_name, operator, reason_text)


_Prev_DashboardService_1 = DashboardService
class DashboardService(_Prev_DashboardService_1):
    def __init__(self, *args, decision_audit_repo: DecisionAuditRepository | None = None, override_lock_repo: OverrideLockRepository | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decision_audit_repo = decision_audit_repo
        self.override_lock_repo = override_lock_repo

    def get_summary(self, business_account_id: int) -> dict:
        summary = super().get_summary(business_account_id)
        if self.decision_audit_repo:
            summary["decision_log_count"] = self.decision_audit_repo.count_for_business(business_account_id)
        if self.override_lock_repo:
            with self.conversation_repo.db.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM user_project_state WHERE business_account_id=%s AND is_locked=TRUE", (business_account_id,))
                row1 = cur.fetchone() or {}
                cur.execute("SELECT COUNT(*) AS cnt FROM user_tags WHERE business_account_id=%s AND is_locked=TRUE AND is_active=TRUE", (business_account_id,))
                row2 = cur.fetchone() or {}
            summary["locked_project_state_count"] = int(row1.get("cnt") or 0)
            summary["locked_tag_count"] = int(row2.get("cnt") or 0)
        summary["pending_review_receipt_count"] = self.receipt_repo.count_pending_receipts(business_account_id)
        return summary


_Prev_ProjectClassifier_1 = ProjectClassifier
class ProjectClassifier(_Prev_ProjectClassifier_1):
    def __init__(self, project_repo: ProjectRepository, override_lock_repo: OverrideLockRepository | None = None) -> None:
        super().__init__(project_repo)
        self.override_lock_repo = override_lock_repo

    def classify(self, business_account_id: int, understanding: UnderstandingResult, context: ConversationContext, user_state: UserStateSnapshot, latest_user_message: str) -> ProjectDecision:
        if self.override_lock_repo and user_state.project_id:
            lock_summary = self.override_lock_repo.get_lock_summary(business_account_id, context.user_id)
            if lock_summary.get("project_locked"):
                return ProjectDecision(project_id=user_state.project_id, confidence=1.0, source="manual_lock", changed=False, reason="project locked by admin")
        return super().classify(business_account_id, understanding, context, user_state, latest_user_message)


class build_admin_blueprint_helper:
    pass


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

    @bp.get("/receipts/pending")
    def pending_receipts():
        business_account_id = request.args.get("business_account_id", type=int)
        limit = request.args.get("limit", default=50, type=int)
        if not business_account_id:
            return jsonify({"ok": False, "error": "business_account_id is required"}), 400
        return jsonify({"ok": True, "data": admin_api_service.list_pending_review_receipts(business_account_id, limit)})

    @bp.get("/customers/<int:business_account_id>/<int:user_id>")
    def customer_detail(business_account_id: int, user_id: int):
        data = admin_api_service.get_customer_detail(business_account_id, user_id)
        return jsonify(data), (200 if data.get("ok") else 404)

    @bp.get("/customers/<int:business_account_id>/<int:user_id>/decision-logs")
    def customer_decision_logs(business_account_id: int, user_id: int):
        limit = request.args.get("limit", default=50, type=int)
        return jsonify({"ok": True, "data": admin_api_service.list_decision_logs(business_account_id, user_id, limit)})

    @bp.post("/customers/<int:business_account_id>/<int:user_id>/project-lock")
    def customer_project_lock(business_account_id: int, user_id: int):
        payload = request.get_json(silent=True) or {}
        data = admin_api_service.lock_project(business_account_id, user_id, payload.get("operator", "admin"), payload.get("reason_text", "manual project lock"))
        return jsonify(data), (200 if data.get("ok") else 400)

    @bp.post("/customers/<int:business_account_id>/<int:user_id>/project-unlock")
    def customer_project_unlock(business_account_id: int, user_id: int):
        payload = request.get_json(silent=True) or {}
        data = admin_api_service.unlock_project(business_account_id, user_id, payload.get("operator", "admin"), payload.get("reason_text", "manual project unlock"))
        return jsonify(data), (200 if data.get("ok") else 400)

    @bp.post("/customers/<int:business_account_id>/<int:user_id>/tags/<string:tag_name>/lock")
    def customer_tag_lock(business_account_id: int, user_id: int, tag_name: str):
        payload = request.get_json(silent=True) or {}
        data = admin_api_service.lock_tag(business_account_id, user_id, tag_name, payload.get("operator", "admin"), payload.get("reason_text", "manual tag lock"))
        return jsonify(data), (200 if data.get("ok") else 400)

    @bp.post("/customers/<int:business_account_id>/<int:user_id>/tags/<string:tag_name>/unlock")
    def customer_tag_unlock(business_account_id: int, user_id: int, tag_name: str):
        payload = request.get_json(silent=True) or {}
        data = admin_api_service.unlock_tag(business_account_id, user_id, tag_name, payload.get("operator", "admin"), payload.get("reason_text", "manual tag unlock"))
        return jsonify(data), (200 if data.get("ok") else 400)

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


def build_app_components(settings: Settings) -> dict[str, Any]:
    db = Database(settings.database_url)
    db.connect()
    initialize_database(db)
    ensure_v1_upgrade_schema(db)
    ensure_v3_upgrade_schema(db)
    ensure_v4_upgrade_schema(db)

    tg_client = TelegramBotAPIClient(
        bot_token=settings.tg_bot_token,
        db=db,
        admin_chat_ids=settings.admin_chat_ids,
    )

    business_account_repo = BusinessAccountRepository(db)
    bootstrap_repo = BootstrapRepository(db)
    conversation_repo = ConversationRepository(db)
    decision_audit_repo = DecisionAuditRepository(db)
    override_lock_repo = OverrideLockRepository(db)
    user_repo = UserRepository(db, decision_audit_repo=decision_audit_repo)
    settings_repo = SettingsRepository(db)
    user_control_repo = UserControlRepository(db)
    material_repo = MaterialRepository(db)
    project_repo = ProjectRepository(db)
    script_repo = ScriptRepository(db)
    receipt_repo = ReceiptRepository(db)
    admin_queue_repo = AdminQueueRepository(db)
    handover_repo = HandoverRepository(db)
    processed_update_repo = ProcessedUpdateRepository(db)
    outbound_message_job_repo = OutboundMessageJobRepository(db)
    followup_repo = FollowupJobRepository(db)

    openai_adapter = OpenAIClientAdapter(settings.openai_api_key)
    llm_service = LLMService(openai_adapter, settings.llm_model_name)
    outbound_queue = OutboundMessageQueue(outbound_message_job_repo, tg_client)
    sender_service = SenderService(TelegramBusinessSenderAdapter(tg_client), outbound_queue=outbound_queue)
    admin_notifier = AdminNotifier(tg_client, admin_chat_ids=settings.admin_chat_ids)
    idempotency_guard = WebhookIdempotencyGuard(processed_update_repo)
    sender_worker = AsyncSenderWorker(settings.database_url, settings.tg_bot_token)
    receipt_center = ReceiptCenter(receipt_repo, decision_audit_repo)

    persona_core = PersonaCore()
    persona_profile_builder = PersonaProfileBuilder(material_repo)
    understanding_engine = UserUnderstandingEngine(llm_service)
    stage_engine = ConversationStageEngine()
    mode_router = ChatModeRouter()
    reply_planner = ReplyPlanner()
    reply_style_engine = ReplyStyleEngine(llm_service)
    reply_self_check_engine = ReplySelfCheckEngine()
    reply_delay_engine = ReplyDelayEngine()
    turn_decision_engine = TurnDecisionEngine()
    humanization_controller = HumanizationController()

    ai_switch_engine = AISwitchEngine(settings_repo, user_control_repo)
    project_classifier = ProjectClassifier(project_repo, override_lock_repo=override_lock_repo)
    project_segment_manager = ProjectSegmentManager(project_repo)
    tagging_engine = TaggingEngine()
    intent_engine = IntentEngine()
    human_escalation_engine = HumanEscalationEngine()
    ops_category_manager = OpsCategoryManager()

    content_selector = ContentSelector(material_repo, ScriptSelector(script_repo), MaterialSelector(material_repo), PersonaMaterialSelector())
    project_nurture_planner = ProjectNurturePlanner()
    memory_writeback_engine = MemoryWritebackEngine(conversation_repo)
    followup_scheduler = FollowupScheduler(followup_repo, outbound_queue, receipt_repo)

    handover_manager = HandoverManager(handover_repo, user_control_repo, conversation_repo, admin_queue_repo)
    handover_summary_builder = HandoverSummaryBuilder(handover_repo, conversation_repo, llm_service)
    resume_chat_manager = ResumeChatManager(handover_repo, conversation_repo, user_control_repo)

    customer_actions = CustomerActions(
        user_control_repo, user_repo, handover_manager, handover_summary_builder,
        resume_chat_manager, handover_repo, conversation_repo,
        receipt_center=receipt_center, override_lock_repo=override_lock_repo,
    )
    admin_api_service = AdminAPIService(
        user_repo, receipt_repo, handover_repo, conversation_repo, user_control_repo,
        admin_queue_repo, customer_actions, resume_chat_manager, project_repo,
        decision_audit_repo=decision_audit_repo, override_lock_repo=override_lock_repo,
    )
    dashboard_service = DashboardService(
        settings_repo, admin_queue_repo, receipt_repo, handover_repo, conversation_repo,
        decision_audit_repo=decision_audit_repo, override_lock_repo=override_lock_repo,
    )
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
        turn_decision_engine=turn_decision_engine,
        humanization_controller=humanization_controller,
        project_nurture_planner=project_nurture_planner,
        memory_writeback_engine=memory_writeback_engine,
        followup_scheduler=followup_scheduler,
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
        "idempotency_guard": idempotency_guard,
        "sender_worker": sender_worker,
        "followup_repo": followup_repo,
        "decision_audit_repo": decision_audit_repo,
        "override_lock_repo": override_lock_repo,
        "receipt_center": receipt_center,
    }

# deferred main call moved to file end

# =========================
# V5 deep-brain upgrade
# =========================

def ensure_v5_upgrade_schema(db: Database) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS strategy_learning_records (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            trajectory TEXT,
            user_style_profile TEXT,
            decision_goal TEXT,
            strategy_notes TEXT,
            outcome_signal TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_style_profiles (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            style_profile TEXT,
            tone_preference TEXT,
            pacing_preference TEXT,
            project_readiness TEXT,
            notes_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (business_account_id, user_id)
        )
        """,
    ]
    with db.transaction():
        with db.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)


class StrategyLearningRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_record(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        trajectory: str,
        user_style_profile: str,
        decision_goal: str,
        strategy_notes: str,
        outcome_signal: str,
    ) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO strategy_learning_records (
                        business_account_id, user_id, conversation_id, trajectory,
                        user_style_profile, decision_goal, strategy_notes, outcome_signal
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        business_account_id,
                        user_id,
                        conversation_id,
                        trajectory,
                        user_style_profile,
                        decision_goal,
                        (strategy_notes or "")[:2000],
                        outcome_signal,
                    ),
                )

    def list_recent_for_user(self, business_account_id: int, user_id: int, limit: int = 15) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM strategy_learning_records
                WHERE business_account_id=%s AND user_id=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (business_account_id, user_id, limit),
            )
            return list(cur.fetchall() or [])


class UserStyleProfileRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_profile(self, business_account_id: int, user_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM user_style_profiles WHERE business_account_id=%s AND user_id=%s LIMIT 1",
                (business_account_id, user_id),
            )
            return cur.fetchone()

    def upsert_profile(
        self,
        business_account_id: int,
        user_id: int,
        style_profile: str,
        tone_preference: str,
        pacing_preference: str,
        project_readiness: str,
        notes_json: dict[str, Any] | None = None,
    ) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_style_profiles (
                        business_account_id, user_id, style_profile, tone_preference,
                        pacing_preference, project_readiness, notes_json, updated_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON CONFLICT (business_account_id, user_id)
                    DO UPDATE SET
                        style_profile=EXCLUDED.style_profile,
                        tone_preference=EXCLUDED.tone_preference,
                        pacing_preference=EXCLUDED.pacing_preference,
                        project_readiness=EXCLUDED.project_readiness,
                        notes_json=EXCLUDED.notes_json,
                        updated_at=NOW()
                    """,
                    (
                        business_account_id,
                        user_id,
                        style_profile,
                        tone_preference,
                        pacing_preference,
                        project_readiness,
                        json.dumps(notes_json or {}, ensure_ascii=False),
                    ),
                )


_Prev_ConversationRepository_2 = ConversationRepository
class ConversationRepository(_Prev_ConversationRepository_2):
    def get_recent_summaries(self, conversation_id: int, limit: int = 3) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT id, summary_text, created_at
                FROM conversation_summaries
                WHERE conversation_id=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (conversation_id, limit),
            )
            return list(cur.fetchall() or [])


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
    emotion_strength: float = 0.4
    boundary_strength: float = 0.0
    social_openness: float = 0.45
    project_relevance: float = 0.0
    human_takeover_hint: bool = False
    notes: list[str] = field(default_factory=list)
    hidden_intent: str | None = None
    conversation_energy: str = "steady"
    hesitation_score: float = 0.0
    style_signal: str = "balanced"
    relationship_readiness: str = "warmup"


@dataclass
class StageDecision:
    stage: str
    changed: bool
    reason: str
    confidence: float = 0.65
    trajectory: str = "steady"
    momentum: str = "neutral"


@dataclass
class TurnDecision:
    reply_goal: str = "rapport"
    should_push_project: bool = False
    social_distance: str = "balanced"
    reply_length: str = "medium"
    ask_followup_question: bool = False
    exit_strategy: str = "gentle_close"
    self_disclosure_level: str = "light"
    reason: str = ""
    should_reply: bool = True
    marketing_intensity: str = "subtle"
    tone_bias: str = "natural"
    need_followup: bool = False
    followup_type: str | None = None
    followup_delay_seconds: int | None = None
    nurture_goal: str | None = None
    should_write_memory: bool = True
    trajectory_goal: str | None = None
    memory_focus: str | None = None
    user_style_profile: str = "balanced"
    project_strategy: str | None = None


@dataclass
class StyleSpec:
    tone: str = "natural"
    formality: str = "balanced"
    warmth: str = "medium"
    length: str = "medium"
    question_rate: str = "low"
    emoji_usage: str = "minimal"
    self_disclosure_ratio: str = "none"
    completion_level: str = "balanced"
    marketing_visibility: str = "subtle"
    naturalness_bias: str = "high"
    cadence_bias: str = "steady"
    initiative_level: str = "low"
    wording_texture: str = "smooth"


class RelationshipTrajectoryEngine:
    def infer(
        self,
        understanding: UnderstandingResult,
        user_state: UserStateSnapshot,
        context: ConversationContext,
        latest_summaries: list[dict] | None = None,
    ) -> tuple[str, str]:
        latest_summaries = latest_summaries or []
        recent_messages = list(context.recent_messages or [])
        text_blob = " ".join((m.get("content_text") or "") for m in recent_messages[-6:]).lower()
        summary_blob = " ".join((s.get("summary_text") or "") for s in latest_summaries).lower()
        trajectory = "steady"
        momentum = "neutral"
        if understanding.boundary_signal or user_state.recent_busy_score > 0.72:
            trajectory, momentum = "cooldown", "down"
        elif understanding.high_intent_signal or user_state.current_project_interest_strength > 0.82:
            trajectory, momentum = "conversion_window", "up"
        elif understanding.explicit_product_query or understanding.project_relevance > 0.72:
            trajectory, momentum = "project_deepen", "up"
        elif "later" in text_blob or "busy" in text_blob or "catch up" in summary_blob:
            trajectory, momentum = "pause_and_return", "flat"
        elif understanding.emotion_state in ("positive", "warm") and understanding.social_openness > 0.62:
            trajectory, momentum = "rapport_building", "up"
        elif understanding.hesitation_score > 0.65 or understanding.resistance_signal:
            trajectory, momentum = "reassure_not_push", "flat"
        return trajectory, momentum


class MemorySelector:
    def select(
        self,
        memory_bundle: MemoryBundle,
        understanding: UnderstandingResult,
        trajectory: str,
    ) -> dict[str, Any]:
        recent_messages = list(memory_bundle.recent_messages or [])
        selected_recent = recent_messages[-8:]
        recent_summary = memory_bundle.recent_summary or ""
        if understanding.boundary_signal:
            selected_recent = recent_messages[-4:]
            recent_summary = (recent_summary or "")[:500]
        elif trajectory in ("conversion_window", "project_deepen"):
            selected_recent = recent_messages[-10:]
        focus = "relationship"
        if trajectory in ("conversion_window", "project_deepen"):
            focus = "project"
        elif trajectory in ("cooldown", "pause_and_return"):
            focus = "boundary"
        return {
            "selected_recent_messages": selected_recent,
            "selected_recent_summary": recent_summary,
            "selected_long_term_memory": memory_bundle.long_term_memory or {},
            "selected_handover_summary": memory_bundle.handover_summary or {},
            "memory_focus": focus,
        }


class AdaptiveStrategyOptimizer:
    def infer_user_style(self, understanding: UnderstandingResult, user_state: UserStateSnapshot) -> str:
        if understanding.boundary_signal or user_state.boundary_sensitivity > 0.75:
            return "space_preferring"
        if understanding.explicit_product_query or user_state.professional_receptiveness > 0.72:
            return "logic_first"
        if understanding.emotion_state in ("low", "anxious") or user_state.emotional_receptiveness > 0.7:
            return "emotion_first"
        if understanding.social_openness > 0.66 and user_state.warmth_score > 0.6:
            return "rapport_first"
        return "balanced"

    def infer_project_readiness(self, trajectory: str, understanding: UnderstandingResult, user_state: UserStateSnapshot) -> str:
        if trajectory == "conversion_window" or understanding.high_intent_signal:
            return "ready"
        if trajectory == "project_deepen" or user_state.current_project_interest_strength > 0.65:
            return "warming"
        if trajectory in ("cooldown", "pause_and_return"):
            return "hold"
        return "early"


class HandoverLearningEngine:
    def __init__(self, strategy_learning_repo: StrategyLearningRepository) -> None:
        self.strategy_learning_repo = strategy_learning_repo

    def learn_after_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        context: ConversationContext,
        turn_decision: TurnDecision,
        stage_decision: StageDecision,
        understanding: UnderstandingResult,
    ) -> None:
        if context.manual_takeover_status not in ("pending_resume", "inactive"):
            return
        if not understanding.human_takeover_hint and turn_decision.trajectory_goal not in ("reassure_then_return", "progress_carefully"):
            return
        notes = (
            f"stage={stage_decision.stage}; trajectory={stage_decision.trajectory}; "
            f"goal={turn_decision.reply_goal}; memory_focus={turn_decision.memory_focus}; "
            f"hidden_intent={understanding.hidden_intent}; style={turn_decision.user_style_profile}"
        )
        self.strategy_learning_repo.insert_record(
            business_account_id,
            user_id,
            conversation_id,
            stage_decision.trajectory,
            turn_decision.user_style_profile,
            turn_decision.reply_goal,
            notes,
            "turn_completed",
        )


_Prev_UserUnderstandingEngine_3 = UserUnderstandingEngine
class UserUnderstandingEngine(_Prev_UserUnderstandingEngine_3):
    def _fallback_rule_based(self, text: str) -> dict:
        result = super()._fallback_rule_based(text)
        t = (text or "").strip().lower()
        hidden_intent = None
        conversation_energy = "steady"
        hesitation_score = 0.2
        style_signal = "balanced"
        relationship_readiness = "warmup"
        notes = list(result.get("notes") or [])
        if any(w in t for w in ["maybe", "not sure", "thinking", "considering", "wondering"]):
            hidden_intent = "curious_but_cautious"
            hesitation_score = 0.68
            notes.append("hesitation_detected")
        if any(w in t for w in ["just curious", "just asking", "no pressure", "not now"]):
            hidden_intent = hidden_intent or "low_pressure_probe"
            hesitation_score = max(hesitation_score, 0.62)
        if any(w in t for w in ["how", "why", "details", "process", "risk"]):
            style_signal = "logic_first"
        elif any(w in t for w in ["feel", "tired", "rough", "long day", "mood"]):
            style_signal = "emotion_first"
        elif any(w in t for w in ["haha", "lol", "nice", "that sounds good", "interesting"]):
            style_signal = "rapport_first"
        if result.get("high_intent_signal"):
            relationship_readiness = "ready_for_progress"
            conversation_energy = "rising"
        elif result.get("boundary_signal"):
            relationship_readiness = "needs_space"
            conversation_energy = "dropping"
        elif result.get("explicit_product_query"):
            relationship_readiness = "ready_for_clarity"
            conversation_energy = "focused"
        elif result.get("emotion_state") in ("positive", "low"):
            conversation_energy = "personal"
        result.update(
            {
                "hidden_intent": hidden_intent,
                "conversation_energy": conversation_energy,
                "hesitation_score": hesitation_score,
                "style_signal": style_signal,
                "relationship_readiness": relationship_readiness,
                "notes": notes,
            }
        )
        return result


_Prev_ConversationStageEngine_4 = ConversationStageEngine
class ConversationStageEngine(_Prev_ConversationStageEngine_4):
    def __init__(self, trajectory_engine: RelationshipTrajectoryEngine | None = None) -> None:
        super().__init__()
        self.trajectory_engine = trajectory_engine or RelationshipTrajectoryEngine()

    def decide(
        self,
        understanding: UnderstandingResult,
        current_stage: str | None,
        user_state: UserStateSnapshot | None = None,
        context: ConversationContext | None = None,
        latest_summaries: list[dict] | None = None,
    ) -> StageDecision:
        user_state = user_state or UserStateSnapshot()
        context = context or ConversationContext([], [], None, None, None, None, None, None, {}, None, None)
        trajectory, momentum = self.trajectory_engine.infer(understanding, user_state, context, latest_summaries)
        base = super().decide(understanding, current_stage)
        target = base.stage
        reason = base.reason
        confidence = getattr(base, "confidence", 0.65)
        if trajectory == "cooldown":
            target = "boundary_protection"
            reason = "trajectory indicates cooldown and space protection"
            confidence = max(confidence, 0.82)
        elif trajectory == "rapport_building" and target not in ("project_discussion", "high_intent_progress"):
            target = "daily_rapport"
            reason = "rapport trajectory is strengthening"
            confidence = max(confidence, 0.74)
        elif trajectory == "project_deepen":
            target = "project_discussion"
            reason = "project discussion is deepening"
            confidence = max(confidence, 0.8)
        elif trajectory == "conversion_window":
            target = "high_intent_progress"
            reason = "conversion window detected"
            confidence = max(confidence, 0.88)
        elif trajectory == "pause_and_return":
            target = "silence_recovery"
            reason = "relationship should recover without pressure"
            confidence = max(confidence, 0.72)
        return StageDecision(
            stage=target,
            changed=(current_stage != target),
            reason=reason,
            confidence=confidence,
            trajectory=trajectory,
            momentum=momentum,
        )


_Prev_TurnDecisionEngine_4 = TurnDecisionEngine
class TurnDecisionEngine(_Prev_TurnDecisionEngine_4):
    def __init__(self, optimizer: AdaptiveStrategyOptimizer | None = None) -> None:
        super().__init__()
        self.optimizer = optimizer or AdaptiveStrategyOptimizer()

    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        user_style_profile = self.optimizer.infer_user_style(understanding, user_state)
        base = super().decide(understanding, stage_decision, mode_decision, user_state)
        trajectory_goal = "maintain_soft_presence"
        memory_focus = "relationship"
        project_strategy = None
        need_followup = base.need_followup
        followup_type = base.followup_type
        followup_delay_seconds = base.followup_delay_seconds
        if stage_decision.trajectory in ("cooldown", "pause_and_return"):
            trajectory_goal = "reassure_then_return"
            memory_focus = "boundary"
            need_followup = False if understanding.boundary_strength > 0.85 else need_followup
            if understanding.boundary_strength < 0.85 and not need_followup:
                need_followup = True
                followup_type = "soft_reconnect"
                followup_delay_seconds = 36 * 3600
            return TurnDecision(
                **{**asdict(base),
                   "reply_goal": "respect_boundary",
                   "social_distance": "give_space",
                   "reply_length": "short",
                   "ask_followup_question": False,
                   "exit_strategy": "leave_space",
                   "marketing_intensity": "none",
                   "tone_bias": "gentle",
                   "trajectory_goal": trajectory_goal,
                   "memory_focus": memory_focus,
                   "user_style_profile": user_style_profile,
                   "project_strategy": None,
                   "need_followup": need_followup,
                   "followup_type": followup_type,
                   "followup_delay_seconds": followup_delay_seconds,
                   "nurture_goal": "protect_trust_not_push",
                   "reason": f"trajectory={stage_decision.trajectory}; style={user_style_profile}"}
            )
        if stage_decision.trajectory == "conversion_window":
            trajectory_goal = "progress_carefully"
            memory_focus = "project"
            project_strategy = "next_step_clarity"
            return TurnDecision(
                **{**asdict(base),
                   "reply_goal": "progress_without_pressure",
                   "should_push_project": True,
                   "reply_length": "medium",
                   "ask_followup_question": user_state.boundary_sensitivity < 0.6,
                   "marketing_intensity": "light",
                   "trajectory_goal": trajectory_goal,
                   "memory_focus": memory_focus,
                   "user_style_profile": user_style_profile,
                   "project_strategy": project_strategy,
                   "need_followup": True,
                   "followup_type": "high_intent_followup",
                   "followup_delay_seconds": 12 * 3600,
                   "nurture_goal": "help_next_step_without_pressure",
                   "reason": f"trajectory={stage_decision.trajectory}; style={user_style_profile}"}
            )
        if stage_decision.trajectory == "project_deepen":
            trajectory_goal = "clarify_and_build_confidence"
            memory_focus = "project"
            project_strategy = "single_clear_point"
            return TurnDecision(
                **{**asdict(base),
                   "reply_goal": "answer_and_transition",
                   "should_push_project": True,
                   "reply_length": "medium",
                   "ask_followup_question": understanding.hesitation_score < 0.55,
                   "trajectory_goal": trajectory_goal,
                   "memory_focus": memory_focus,
                   "user_style_profile": user_style_profile,
                   "project_strategy": project_strategy,
                   "need_followup": True if understanding.explicit_product_query else base.need_followup,
                   "followup_type": "light_project_followup" if understanding.explicit_product_query else base.followup_type,
                   "followup_delay_seconds": 20 * 3600 if understanding.explicit_product_query else base.followup_delay_seconds,
                   "nurture_goal": "reduce_friction_and_keep_clarity",
                   "reason": f"trajectory={stage_decision.trajectory}; style={user_style_profile}"}
            )
        if user_style_profile == "emotion_first":
            trajectory_goal = "warm_and_ground"
            return TurnDecision(
                **{**asdict(base),
                   "reply_goal": "support_first" if understanding.emotion_state in ("low", "anxious") else base.reply_goal,
                   "marketing_intensity": "none" if understanding.emotion_state in ("low", "anxious") else base.marketing_intensity,
                   "self_disclosure_level": "light",
                   "trajectory_goal": trajectory_goal,
                   "memory_focus": "relationship",
                   "user_style_profile": user_style_profile,
                   "reason": f"emotion-first style; trajectory={stage_decision.trajectory}"}
            )
        if user_style_profile == "logic_first":
            trajectory_goal = "clear_and_structured"
            return TurnDecision(
                **{**asdict(base),
                   "reply_length": "medium",
                   "tone_bias": "clear",
                   "self_disclosure_level": "none",
                   "trajectory_goal": trajectory_goal,
                   "memory_focus": "project" if understanding.project_relevance > 0.55 else "relationship",
                   "user_style_profile": user_style_profile,
                   "project_strategy": "single_clear_point" if base.should_push_project else None,
                   "reason": f"logic-first style; trajectory={stage_decision.trajectory}"}
            )
        return TurnDecision(
            **{**asdict(base),
               "trajectory_goal": trajectory_goal,
               "memory_focus": memory_focus,
               "user_style_profile": user_style_profile,
               "project_strategy": project_strategy,
               "reason": f"trajectory={stage_decision.trajectory}; style={user_style_profile}; base={base.reason}"}
        )


_Prev_HumanizationController_3 = HumanizationController
class HumanizationController(_Prev_HumanizationController_3):
    def build_style_spec(
        self,
        decision: TurnDecision,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        persona_profile: dict[str, Any] | None = None,
    ) -> StyleSpec:
        base = super().build_style_spec(decision, understanding, stage_decision, user_state, persona_profile)
        cadence_bias = "steady"
        initiative_level = "low"
        wording_texture = "smooth"
        if decision.user_style_profile == "logic_first":
            wording_texture = "clean"
            cadence_bias = "measured"
        elif decision.user_style_profile == "emotion_first":
            wording_texture = "gentle"
            cadence_bias = "soft"
        elif decision.user_style_profile == "space_preferring":
            wording_texture = "minimal"
            cadence_bias = "slow"
            initiative_level = "minimal"
        elif decision.user_style_profile == "rapport_first":
            wording_texture = "warm"
            cadence_bias = "light"
            initiative_level = "soft"
        if stage_decision.trajectory in ("conversion_window", "project_deepen"):
            initiative_level = "moderate" if decision.ask_followup_question else "soft"
        elif stage_decision.trajectory in ("cooldown", "pause_and_return"):
            initiative_level = "minimal"
        return StyleSpec(
            tone=base.tone,
            formality=base.formality,
            warmth=base.warmth,
            length=base.length,
            question_rate=base.question_rate,
            emoji_usage=base.emoji_usage,
            self_disclosure_ratio=base.self_disclosure_ratio,
            completion_level=base.completion_level,
            marketing_visibility=base.marketing_visibility,
            naturalness_bias=base.naturalness_bias,
            cadence_bias=cadence_bias,
            initiative_level=initiative_level,
            wording_texture=wording_texture,
        )


_Prev_ProjectNurturePlanner_1 = ProjectNurturePlanner
class ProjectNurturePlanner(_Prev_ProjectNurturePlanner_1):
    def plan(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        turn_decision: TurnDecision,
        project_decision: ProjectDecision,
        user_state: UserStateSnapshot,
    ) -> dict[str, Any]:
        base = super().plan(understanding, stage_decision, turn_decision, project_decision, user_state)
        if not base.get("should_include_project_content"):
            return base
        angle = turn_decision.project_strategy or base.get("nurture_angle") or "single_clear_point"
        max_points = base.get("max_project_points") or 1
        if understanding.hesitation_score > 0.6:
            angle = "reduce_pressure_then_clarify"
            max_points = 1
        if turn_decision.user_style_profile == "logic_first":
            angle = "clear_fact_then_soft_bridge"
        elif turn_decision.user_style_profile == "emotion_first":
            angle = "reassure_then_light_value"
            max_points = 1
        return {
            **base,
            "nurture_angle": angle,
            "max_project_points": max_points,
            "trajectory_goal": turn_decision.trajectory_goal,
        }


_Prev_MemoryWritebackEngine_1 = MemoryWritebackEngine
class MemoryWritebackEngine(_Prev_MemoryWritebackEngine_1):
    def __init__(self, conversation_repo: ConversationRepository, style_profile_repo: UserStyleProfileRepository | None = None, strategy_learning_repo: StrategyLearningRepository | None = None) -> None:
        super().__init__(conversation_repo)
        self.style_profile_repo = style_profile_repo
        self.strategy_learning_repo = strategy_learning_repo

    def write_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        turn_decision: TurnDecision,
        stage_decision: StageDecision,
        understanding: UnderstandingResult,
        final_reply_text: str,
        latest_user_text: str,
    ) -> None:
        super().write_turn(business_account_id, user_id, conversation_id, turn_decision, stage_decision, understanding, final_reply_text, latest_user_text)
        if self.style_profile_repo:
            self.style_profile_repo.upsert_profile(
                business_account_id,
                user_id,
                turn_decision.user_style_profile,
                turn_decision.tone_bias,
                "slow" if turn_decision.social_distance == "give_space" else "normal",
                "ready" if stage_decision.trajectory in ("conversion_window", "project_deepen") else "warming",
                notes_json={
                    "memory_focus": turn_decision.memory_focus,
                    "trajectory_goal": turn_decision.trajectory_goal,
                    "relationship_readiness": understanding.relationship_readiness,
                },
            )
        if self.strategy_learning_repo:
            self.strategy_learning_repo.insert_record(
                business_account_id,
                user_id,
                conversation_id,
                stage_decision.trajectory,
                turn_decision.user_style_profile,
                turn_decision.reply_goal,
                f"memory_focus={turn_decision.memory_focus}; tone={turn_decision.tone_bias}; exit={turn_decision.exit_strategy}",
                "reply_written",
            )


_Prev_Orchestrator_5 = Orchestrator
class Orchestrator(_Prev_Orchestrator_5):
    def __init__(
        self,
        *args,
        memory_selector: MemorySelector | None = None,
        trajectory_engine: RelationshipTrajectoryEngine | None = None,
        strategy_optimizer: AdaptiveStrategyOptimizer | None = None,
        strategy_learning_repo: StrategyLearningRepository | None = None,
        style_profile_repo: UserStyleProfileRepository | None = None,
        handover_learning_engine: HandoverLearningEngine | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory_selector = memory_selector or MemorySelector()
        self.trajectory_engine = trajectory_engine or RelationshipTrajectoryEngine()
        self.strategy_optimizer = strategy_optimizer or AdaptiveStrategyOptimizer()
        self.strategy_learning_repo = strategy_learning_repo
        self.style_profile_repo = style_profile_repo
        self.handover_learning_engine = handover_learning_engine

    def _build_memory_bundle(self, context: ConversationContext, latest_handover_summary: dict[str, Any] | None = None) -> MemoryBundle:
        bundle = super()._build_memory_bundle(context, latest_handover_summary)
        summaries = []
        if getattr(context, "conversation_id", None):
            try:
                summaries = self.conversation_repo.get_recent_summaries(context.conversation_id, 3)
            except Exception:
                summaries = []
        if summaries:
            joined = " | ".join((item.get("summary_text") or "")[:260] for item in summaries)
            bundle.recent_summary = (bundle.recent_summary + " | " + joined).strip(" |")
        return bundle

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
        try:
            setattr(context, "conversation_id", conversation_id)
        except Exception:
            pass
        user_state = self.user_repo.get_user_state_snapshot(business_account_id, user_id)
        ai_allowed, ai_reason = self.ai_switch_engine.decide(business_account_id, conversation_id, user_state.ops_category, context.manual_takeover_status)
        if not ai_allowed:
            logger.info("AI reply skipped | conversation_id=%s | reason=%s", conversation_id, ai_reason)
            return

        latest_user_text = (inbound_message.text or "").strip()
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
        memory_bundle = self._build_memory_bundle(context, latest_handover_summary)
        latest_summaries = self.conversation_repo.get_recent_summaries(conversation_id, 3)
        understanding = self.understanding_engine.analyze(latest_user_text, recent_context, self.persona_core.to_summary(), self._user_state_summary(user_state))
        stage_decision = self.stage_engine.decide(understanding, context.current_stage, user_state=user_state, context=context, latest_summaries=latest_summaries)
        mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
        turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
        selected_memory = self.memory_selector.select(memory_bundle, understanding, stage_decision.trajectory)
        style_spec = self.humanization_controller.build_style_spec(turn_decision, understanding, stage_decision, user_state, persona_profile)
        project_decision = self.project_classifier.classify(business_account_id, understanding, context, user_state, latest_user_text)
        intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
        segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
        escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
        escalation_decision["should_queue_admin"] = bool(escalation_decision.get("should_queue_admin") or understanding.human_takeover_hint)
        if understanding.human_takeover_hint and not escalation_decision.get("reason"):
            escalation_decision["reason"] = "user asked for a real person or direct explanation"
            escalation_decision["notify_level"] = "suggest_takeover"
        tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
        ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)
        reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
        reply_plan.should_reply = turn_decision.should_reply
        reply_plan.should_continue_product = turn_decision.should_push_project
        reply_plan.should_leave_space = turn_decision.exit_strategy in ("leave_space", "soft_hold")
        reply_plan.should_self_share = turn_decision.self_disclosure_level != "none"
        selected_content = self.content_selector.select(business_account_id, project_decision.project_id, mode_decision.chat_mode, reply_plan)
        if hasattr(self, "project_nurture_planner") and self.project_nurture_planner:
            selected_content["v5_project_nurture"] = self.project_nurture_planner.plan(
                understanding, stage_decision, turn_decision, project_decision, user_state
            )
        if not turn_decision.should_push_project and not understanding.explicit_product_query:
            selected_content = {k: v for k, v in (selected_content or {}).items() if k in ("persona_materials", "self_share_materials", "v5_project_nurture")}
        understanding_payload = understanding.__dict__.copy()
        understanding_payload["turn_decision"] = asdict(turn_decision)
        understanding_payload["style_spec"] = asdict(style_spec)
        understanding_payload["memory_summary"] = {
            "recent_messages_count": len(selected_memory.get("selected_recent_messages") or []),
            "has_recent_summary": bool(selected_memory.get("selected_recent_summary")),
            "has_handover_summary": bool(selected_memory.get("selected_handover_summary")),
            "memory_focus": selected_memory.get("memory_focus"),
            "trajectory": stage_decision.trajectory,
        }
        if latest_handover_summary:
            understanding_payload["resume_hint"] = latest_handover_summary.get("resume_suggestion")
        draft_reply = self.reply_style_engine.generate(
            latest_user_text,
            selected_memory.get("selected_recent_messages") or recent_context,
            persona_summary,
            self._user_state_summary(user_state),
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding_payload,
            {**reply_plan.__dict__, "trajectory_goal": turn_decision.trajectory_goal, "memory_focus": turn_decision.memory_focus},
            selected_content,
        )
        final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        final_reply = FinalReply(
            text=final_text,
            delay_seconds=delay_seconds,
            metadata={"turn_decision": asdict(turn_decision), "style_spec": asdict(style_spec), "trajectory": stage_decision.trajectory},
        )
        if reply_plan.should_reply and final_reply.text.strip():
            send_result = self.sender_service.send_text_reply(
                conversation_id,
                final_reply.text,
                final_reply.delay_seconds,
                raw_payload=inbound_message.raw_payload,
                metadata={
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "trajectory": stage_decision.trajectory,
                    "delivery": "queued",
                },
            )
            self.conversation_repo.save_message(
                conversation_id,
                "ai",
                "text",
                final_reply.text,
                {
                    "selected_content": selected_content,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "trajectory": stage_decision.trajectory,
                    "delivery": send_result,
                },
                None,
            )
            self.conversation_repo.set_last_ai_reply_at(conversation_id)
        self.conversation_repo.update_conversation_state(conversation_id, stage_decision.stage, mode_decision.chat_mode, understanding.current_mainline_should_continue)
        if project_decision.project_id is not None:
            self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
        self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_decision["ops_category"], ops_decision["reason"], "system")
        try:
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "AI Turn Decision",
                {
                    "conversation_id": conversation_id,
                    "stage": stage_decision.stage,
                    "stage_confidence": stage_decision.confidence,
                    "trajectory": stage_decision.trajectory,
                    "chat_mode": mode_decision.chat_mode,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "intent_level": intent_decision.level,
                    "intent_score": intent_decision.score,
                },
            )
        except Exception:
            logger.exception("Failed to create AI turn-decision receipt")
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
                    "style_spec": asdict(style_spec),
                    "trajectory": stage_decision.trajectory,
                },
            )
            if self.admin_notifier:
                self.admin_notifier.notify_high_intent(business_account_id, user_id, conversation_id, escalation_decision.get("notify_level") or "watch", escalation_decision.get("reason") or "")
        if hasattr(self, "memory_writeback_engine") and self.memory_writeback_engine and turn_decision.should_write_memory:
            try:
                self.memory_writeback_engine.write_turn(
                    business_account_id,
                    user_id,
                    conversation_id,
                    turn_decision,
                    stage_decision,
                    understanding,
                    final_reply.text,
                    latest_user_text,
                )
            except Exception:
                logger.exception("V5 memory writeback failed")
        if self.handover_learning_engine:
            try:
                self.handover_learning_engine.learn_after_turn(
                    business_account_id,
                    user_id,
                    conversation_id,
                    context,
                    turn_decision,
                    stage_decision,
                    understanding,
                )
            except Exception:
                logger.exception("V5 handover learning failed")
        if hasattr(self, "followup_scheduler") and self.followup_scheduler and turn_decision.need_followup:
            try:
                self.followup_scheduler.schedule(
                    business_account_id,
                    user_id,
                    conversation_id,
                    turn_decision,
                    understanding,
                    stage_decision,
                )
            except Exception:
                logger.exception("V5 followup scheduling failed")


def build_app_components(settings: Settings) -> dict[str, Any]:
    db = Database(settings.database_url)
    db.connect()
    initialize_database(db)
    ensure_v1_upgrade_schema(db)
    ensure_v3_upgrade_schema(db)
    ensure_v4_upgrade_schema(db)
    ensure_v5_upgrade_schema(db)

    tg_client = TelegramBotAPIClient(
        bot_token=settings.tg_bot_token,
        db=db,
        admin_chat_ids=settings.admin_chat_ids,
    )

    business_account_repo = BusinessAccountRepository(db)
    bootstrap_repo = BootstrapRepository(db)
    conversation_repo = ConversationRepository(db)
    decision_audit_repo = DecisionAuditRepository(db)
    override_lock_repo = OverrideLockRepository(db)
    strategy_learning_repo = StrategyLearningRepository(db)
    style_profile_repo = UserStyleProfileRepository(db)
    user_repo = UserRepository(db, decision_audit_repo=decision_audit_repo)
    settings_repo = SettingsRepository(db)
    user_control_repo = UserControlRepository(db)
    material_repo = MaterialRepository(db)
    project_repo = ProjectRepository(db)
    script_repo = ScriptRepository(db)
    receipt_repo = ReceiptRepository(db)
    admin_queue_repo = AdminQueueRepository(db)
    handover_repo = HandoverRepository(db)
    processed_update_repo = ProcessedUpdateRepository(db)
    outbound_message_job_repo = OutboundMessageJobRepository(db)
    followup_repo = FollowupJobRepository(db)

    openai_adapter = OpenAIClientAdapter(settings.openai_api_key)
    llm_service = LLMService(openai_adapter, settings.llm_model_name)
    outbound_queue = OutboundMessageQueue(outbound_message_job_repo, tg_client)
    sender_service = SenderService(TelegramBusinessSenderAdapter(tg_client), outbound_queue=outbound_queue)
    admin_notifier = AdminNotifier(tg_client, admin_chat_ids=settings.admin_chat_ids)
    idempotency_guard = WebhookIdempotencyGuard(processed_update_repo)
    sender_worker = AsyncSenderWorker(settings.database_url, settings.tg_bot_token)
    receipt_center = ReceiptCenter(receipt_repo, decision_audit_repo)

    persona_core = PersonaCore()
    persona_profile_builder = PersonaProfileBuilder(material_repo)
    understanding_engine = UserUnderstandingEngine(llm_service)
    trajectory_engine = RelationshipTrajectoryEngine()
    stage_engine = ConversationStageEngine(trajectory_engine=trajectory_engine)
    mode_router = ChatModeRouter()
    reply_planner = ReplyPlanner()
    reply_style_engine = ReplyStyleEngine(llm_service)
    reply_self_check_engine = ReplySelfCheckEngine()
    reply_delay_engine = ReplyDelayEngine()
    strategy_optimizer = AdaptiveStrategyOptimizer()
    turn_decision_engine = TurnDecisionEngine(optimizer=strategy_optimizer)
    humanization_controller = HumanizationController()

    ai_switch_engine = AISwitchEngine(settings_repo, user_control_repo)
    project_classifier = ProjectClassifier(project_repo, override_lock_repo=override_lock_repo)
    project_segment_manager = ProjectSegmentManager(project_repo)
    tagging_engine = TaggingEngine()
    intent_engine = IntentEngine()
    human_escalation_engine = HumanEscalationEngine()
    ops_category_manager = OpsCategoryManager()

    content_selector = ContentSelector(material_repo, ScriptSelector(script_repo), MaterialSelector(material_repo), PersonaMaterialSelector())
    project_nurture_planner = ProjectNurturePlanner()
    memory_writeback_engine = MemoryWritebackEngine(conversation_repo, style_profile_repo=style_profile_repo, strategy_learning_repo=strategy_learning_repo)
    followup_scheduler = FollowupScheduler(followup_repo, outbound_queue, receipt_repo)
    handover_learning_engine = HandoverLearningEngine(strategy_learning_repo)

    handover_manager = HandoverManager(handover_repo, user_control_repo, conversation_repo, admin_queue_repo)
    handover_summary_builder = HandoverSummaryBuilder(handover_repo, conversation_repo, llm_service)
    resume_chat_manager = ResumeChatManager(handover_repo, conversation_repo, user_control_repo)

    customer_actions = CustomerActions(
        user_control_repo, user_repo, handover_manager, handover_summary_builder,
        resume_chat_manager, handover_repo, conversation_repo,
        receipt_center=receipt_center, override_lock_repo=override_lock_repo,
    )
    admin_api_service = AdminAPIService(
        user_repo, receipt_repo, handover_repo, conversation_repo, user_control_repo,
        admin_queue_repo, customer_actions, resume_chat_manager, project_repo,
        decision_audit_repo=decision_audit_repo, override_lock_repo=override_lock_repo,
    )
    dashboard_service = DashboardService(
        settings_repo, admin_queue_repo, receipt_repo, handover_repo, conversation_repo,
        decision_audit_repo=decision_audit_repo, override_lock_repo=override_lock_repo,
    )
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
        turn_decision_engine=turn_decision_engine,
        humanization_controller=humanization_controller,
        project_nurture_planner=project_nurture_planner,
        memory_writeback_engine=memory_writeback_engine,
        followup_scheduler=followup_scheduler,
        memory_selector=MemorySelector(),
        trajectory_engine=trajectory_engine,
        strategy_optimizer=strategy_optimizer,
        strategy_learning_repo=strategy_learning_repo,
        style_profile_repo=style_profile_repo,
        handover_learning_engine=handover_learning_engine,
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
        "idempotency_guard": idempotency_guard,
        "sender_worker": sender_worker,
        "followup_repo": followup_repo,
        "decision_audit_repo": decision_audit_repo,
        "override_lock_repo": override_lock_repo,
        "receipt_center": receipt_center,
        "strategy_learning_repo": strategy_learning_repo,
        "style_profile_repo": style_profile_repo,
    }


# =========================
# V5 continued deep-brain optimization
# =========================

@dataclass
class TurnDecision:
    reply_goal: str = "rapport"
    should_push_project: bool = False
    social_distance: str = "balanced"
    reply_length: str = "medium"
    ask_followup_question: bool = False
    exit_strategy: str = "gentle_close"
    self_disclosure_level: str = "light"
    reason: str = ""
    should_reply: bool = True
    marketing_intensity: str = "subtle"
    tone_bias: str = "natural"
    need_followup: bool = False
    followup_type: str | None = None
    followup_delay_seconds: int | None = None
    nurture_goal: str | None = None
    should_write_memory: bool = True
    trajectory_goal: str | None = None
    memory_focus: str | None = None
    user_style_profile: str = "balanced"
    project_strategy: str | None = None
    relationship_rhythm: str = "steady"
    engagement_mode: str = "maintain"
    recovery_window_hours: int = 24
    trust_signal: str = "neutral"


@dataclass
class StyleSpec:
    tone: str = "natural"
    formality: str = "balanced"
    warmth: str = "medium"
    length: str = "medium"
    question_rate: str = "low"
    emoji_usage: str = "minimal"
    self_disclosure_ratio: str = "none"
    completion_level: str = "balanced"
    marketing_visibility: str = "subtle"
    naturalness_bias: str = "high"
    cadence_bias: str = "steady"
    initiative_level: str = "low"
    wording_texture: str = "smooth"
    pause_texture: str = "natural"
    anchor_style: str = "light"


class RelationshipRhythmEngine:
    def decide(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        stored_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stored_profile = stored_profile or {}
        style_profile = (stored_profile.get("style_profile") or "").lower()
        rhythm = "steady"
        engagement_mode = "maintain"
        recovery_window_hours = 24
        trust_signal = "neutral"

        if stage_decision.trajectory in ("cooldown", "pause_and_return") or understanding.boundary_signal:
            rhythm = "slow"
            engagement_mode = "protect_space"
            recovery_window_hours = 36 if understanding.boundary_strength < 0.85 else 60
            trust_signal = "fragile"
        elif stage_decision.trajectory in ("conversion_window", "project_deepen"):
            rhythm = "responsive"
            engagement_mode = "clarify_and_progress"
            recovery_window_hours = 12
            trust_signal = "ready"
        elif understanding.hesitation_score > 0.65:
            rhythm = "patient"
            engagement_mode = "reassure_and_hold"
            recovery_window_hours = 30
            trust_signal = "building"
        elif understanding.emotion_state in ("positive", "warm") and understanding.social_openness > 0.62:
            rhythm = "warm"
            engagement_mode = "bond_and_expand"
            recovery_window_hours = 20
            trust_signal = "opening"

        if style_profile == "space_preferring":
            rhythm = "slow"
            recovery_window_hours = max(recovery_window_hours, 30)
        elif style_profile == "logic_first" and engagement_mode == "clarify_and_progress":
            rhythm = "measured"
        elif style_profile == "emotion_first" and engagement_mode not in ("clarify_and_progress", "protect_space"):
            rhythm = "soft"

        return {
            "relationship_rhythm": rhythm,
            "engagement_mode": engagement_mode,
            "recovery_window_hours": recovery_window_hours,
            "trust_signal": trust_signal,
        }


_Prev_AdaptiveStrategyOptimizer_1 = AdaptiveStrategyOptimizer
class AdaptiveStrategyOptimizer(_Prev_AdaptiveStrategyOptimizer_1):
    def infer_user_style(
        self,
        understanding: UnderstandingResult,
        user_state: UserStateSnapshot,
        stored_profile: dict[str, Any] | None = None,
        recent_learning: list[dict] | None = None,
    ) -> str:
        stored_profile = stored_profile or {}
        recent_learning = recent_learning or []
        stored_style = stored_profile.get("style_profile")
        if stored_style:
            if stored_style == "space_preferring" and not understanding.high_intent_signal:
                return "space_preferring"
            if stored_style == "logic_first" and understanding.style_signal in ("balanced", "logic_first"):
                return "logic_first"
            if stored_style == "emotion_first" and understanding.style_signal in ("balanced", "emotion_first"):
                return "emotion_first"

        recent_styles = [r.get("user_style_profile") for r in recent_learning if r.get("user_style_profile")]
        if recent_styles:
            dominant = max(set(recent_styles), key=recent_styles.count)
            if dominant and dominant != "balanced" and understanding.style_signal == "balanced":
                return dominant

        return super().infer_user_style(understanding, user_state)

    def infer_anchor_style(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_style_profile: str,
    ) -> str:
        if stage_decision.trajectory in ("cooldown", "pause_and_return"):
            return "reassuring"
        if user_style_profile == "logic_first":
            return "clarity_first"
        if user_style_profile == "emotion_first":
            return "empathy_first"
        if understanding.hesitation_score > 0.62:
            return "low_pressure"
        return "light"


_Prev_MemorySelector_1 = MemorySelector
class MemorySelector(_Prev_MemorySelector_1):
    def select(
        self,
        memory_bundle: MemoryBundle,
        understanding: UnderstandingResult,
        trajectory: str,
        stored_profile: dict[str, Any] | None = None,
        recent_learning: list[dict] | None = None,
    ) -> dict[str, Any]:
        base = super().select(memory_bundle, understanding, trajectory)
        stored_profile = stored_profile or {}
        recent_learning = recent_learning or []
        selected_recent = list(base.get("selected_recent_messages") or [])
        selected_summary = base.get("selected_recent_summary") or ""
        focus = base.get("memory_focus") or "relationship"
        strategy_context = []

        style_profile = stored_profile.get("style_profile") or "balanced"
        if style_profile == "space_preferring":
            selected_recent = selected_recent[-3:]
            selected_summary = (selected_summary or "")[:360]
        elif style_profile == "logic_first":
            selected_recent = selected_recent[-6:]
            focus = "project" if trajectory in ("conversion_window", "project_deepen") else focus
        elif style_profile == "emotion_first":
            selected_recent = selected_recent[-5:]

        if understanding.hesitation_score > 0.62:
            focus = "reassurance"
            selected_summary = (selected_summary + " | caution: user may need low-pressure clarity").strip(" |")
        if understanding.boundary_signal:
            focus = "boundary"
        elif understanding.high_intent_signal and trajectory == "conversion_window":
            focus = "project_clarity"

        for item in recent_learning[:3]:
            note = item.get("strategy_notes")
            if note:
                strategy_context.append(str(note)[:240])

        return {
            **base,
            "selected_recent_messages": selected_recent,
            "selected_recent_summary": selected_summary,
            "memory_focus": focus,
            "stored_style_profile": style_profile,
            "strategy_context": strategy_context,
        }


_Prev_TurnDecisionEngine_5 = TurnDecisionEngine
class TurnDecisionEngine(_Prev_TurnDecisionEngine_5):
    def __init__(
        self,
        optimizer: AdaptiveStrategyOptimizer | None = None,
        rhythm_engine: RelationshipRhythmEngine | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer or AdaptiveStrategyOptimizer())
        self.optimizer = optimizer or AdaptiveStrategyOptimizer()
        self.rhythm_engine = rhythm_engine or RelationshipRhythmEngine()

    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        stored_profile = getattr(user_state, "_style_profile_data", None) or {}
        recent_learning = getattr(user_state, "_recent_strategy_learning", None) or []
        user_style_profile = self.optimizer.infer_user_style(
            understanding,
            user_state,
            stored_profile=stored_profile,
            recent_learning=recent_learning,
        )
        base = super().decide(understanding, stage_decision, mode_decision, user_state)
        rhythm = self.rhythm_engine.decide(understanding, stage_decision, user_state, stored_profile=stored_profile)
        anchor_style = self.optimizer.infer_anchor_style(understanding, stage_decision, user_style_profile)

        data = {**asdict(base)}
        data["user_style_profile"] = user_style_profile
        data["relationship_rhythm"] = rhythm["relationship_rhythm"]
        data["engagement_mode"] = rhythm["engagement_mode"]
        data["recovery_window_hours"] = rhythm["recovery_window_hours"]
        data["trust_signal"] = rhythm["trust_signal"]

        if rhythm["engagement_mode"] == "protect_space":
            data.update(
                {
                    "reply_goal": "respect_boundary",
                    "social_distance": "give_space",
                    "reply_length": "short",
                    "ask_followup_question": False,
                    "exit_strategy": "leave_space",
                    "marketing_intensity": "none",
                    "tone_bias": "gentle",
                    "trajectory_goal": "protect_and_return_later",
                    "memory_focus": "boundary",
                    "nurture_goal": "protect_trust_not_push",
                    "project_strategy": None,
                    "followup_delay_seconds": max(data.get("followup_delay_seconds") or 0, rhythm["recovery_window_hours"] * 3600) if data.get("need_followup") else rhythm["recovery_window_hours"] * 3600,
                    "reason": f"trajectory={stage_decision.trajectory}; rhythm={rhythm['relationship_rhythm']}; anchor={anchor_style}",
                }
            )
            data["need_followup"] = False if understanding.boundary_strength > 0.85 else bool(data.get("need_followup"))
            data["followup_type"] = None if not data["need_followup"] else (data.get("followup_type") or "soft_reconnect")
            return TurnDecision(**data)

        if rhythm["engagement_mode"] == "clarify_and_progress":
            data.update(
                {
                    "reply_goal": "progress_with_clarity",
                    "social_distance": "warm_professional",
                    "reply_length": "medium" if user_style_profile == "logic_first" else "medium_long",
                    "ask_followup_question": True,
                    "marketing_intensity": "light",
                    "tone_bias": "clear",
                    "trajectory_goal": "clarify_then_progress",
                    "memory_focus": "project_clarity",
                    "nurture_goal": "reduce_friction_and_next_step",
                    "project_strategy": "next_step_clarity" if understanding.hesitation_score < 0.55 else "reduce_pressure_then_clarify",
                    "need_followup": True,
                    "followup_type": data.get("followup_type") or "high_intent_followup",
                    "followup_delay_seconds": min(data.get("followup_delay_seconds") or 18 * 3600, rhythm["recovery_window_hours"] * 3600),
                    "reason": f"trajectory={stage_decision.trajectory}; rhythm={rhythm['relationship_rhythm']}; anchor={anchor_style}",
                }
            )
            return TurnDecision(**data)

        if rhythm["engagement_mode"] == "reassure_and_hold":
            data.update(
                {
                    "reply_goal": "reassure_without_pressure",
                    "social_distance": "gentle",
                    "reply_length": "short" if user_style_profile == "space_preferring" else "medium",
                    "ask_followup_question": False if understanding.hesitation_score > 0.72 else data.get("ask_followup_question"),
                    "marketing_intensity": "none",
                    "tone_bias": "reassuring",
                    "trajectory_goal": "build_safety_before_progress",
                    "memory_focus": "reassurance",
                    "nurture_goal": "reduce_friction_and_keep_warm",
                    "project_strategy": "light_value_bridge" if understanding.project_relevance > 0.5 else None,
                    "need_followup": True if understanding.project_relevance > 0.38 and not understanding.boundary_signal else bool(data.get("need_followup")),
                    "followup_type": data.get("followup_type") or ("soft_reconnect" if understanding.project_relevance > 0.38 else None),
                    "followup_delay_seconds": rhythm["recovery_window_hours"] * 3600,
                    "reason": f"trajectory={stage_decision.trajectory}; hesitation={understanding.hesitation_score:.2f}; anchor={anchor_style}",
                }
            )
            return TurnDecision(**data)

        data.update(
            {
                "trajectory_goal": data.get("trajectory_goal") or "maintain_and_warm",
                "memory_focus": data.get("memory_focus") or ("relationship" if stage_decision.stage in ("first_contact", "daily_rapport", "trust_building") else "project"),
                "reason": data.get("reason") or f"trajectory={stage_decision.trajectory}; rhythm={rhythm['relationship_rhythm']}; anchor={anchor_style}",
            }
        )
        return TurnDecision(**data)


_Prev_HumanizationController_4 = HumanizationController
class HumanizationController(_Prev_HumanizationController_4):
    def __init__(self, optimizer: AdaptiveStrategyOptimizer | None = None) -> None:
        super().__init__()
        self.optimizer = optimizer or AdaptiveStrategyOptimizer()

    def build_style_spec(
        self,
        decision: TurnDecision,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        persona_profile: dict[str, Any] | None = None,
    ) -> StyleSpec:
        base = super().build_style_spec(decision, understanding, stage_decision, user_state, persona_profile)
        pause_texture = "natural"
        anchor_style = self.optimizer.infer_anchor_style(understanding, stage_decision, decision.user_style_profile)
        cadence_bias = base.cadence_bias
        initiative_level = base.initiative_level
        wording_texture = base.wording_texture

        if decision.relationship_rhythm in ("slow", "patient"):
            pause_texture = "soft_pause"
            cadence_bias = "slow"
            initiative_level = "minimal"
        elif decision.relationship_rhythm in ("responsive", "warm"):
            pause_texture = "engaged"
            cadence_bias = "attentive" if decision.user_style_profile == "logic_first" else "light"
            initiative_level = "soft" if decision.user_style_profile != "space_preferring" else "minimal"

        if anchor_style == "clarity_first":
            wording_texture = "clean"
        elif anchor_style == "empathy_first":
            wording_texture = "gentle"
        elif anchor_style == "reassuring":
            wording_texture = "calm"
        elif anchor_style == "low_pressure":
            wording_texture = "minimal"

        return StyleSpec(
            tone=base.tone,
            formality=base.formality,
            warmth=base.warmth,
            length=base.length,
            question_rate=base.question_rate,
            emoji_usage=base.emoji_usage,
            self_disclosure_ratio=base.self_disclosure_ratio,
            completion_level=base.completion_level,
            marketing_visibility=base.marketing_visibility,
            naturalness_bias=base.naturalness_bias,
            cadence_bias=cadence_bias,
            initiative_level=initiative_level,
            wording_texture=wording_texture,
            pause_texture=pause_texture,
            anchor_style=anchor_style,
        )


_Prev_ProjectNurturePlanner_2 = ProjectNurturePlanner
class ProjectNurturePlanner(_Prev_ProjectNurturePlanner_2):
    def plan(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        turn_decision: TurnDecision,
        project_decision: ProjectDecision,
        user_state: UserStateSnapshot,
    ) -> dict[str, Any]:
        base = super().plan(understanding, stage_decision, turn_decision, project_decision, user_state)
        if not base.get("should_include_project_content"):
            return base

        angle = base.get("nurture_angle") or turn_decision.project_strategy or "single_clear_point"
        max_points = int(base.get("max_project_points") or 1)

        if turn_decision.engagement_mode == "reassure_and_hold":
            angle = "reduce_pressure_then_clarify"
            max_points = 1
        elif turn_decision.user_style_profile == "logic_first":
            angle = "clear_fact_then_soft_bridge"
            max_points = 2 if turn_decision.relationship_rhythm == "responsive" else 1
        elif turn_decision.user_style_profile == "emotion_first":
            angle = "reassure_then_light_value"
            max_points = 1
        elif turn_decision.user_style_profile == "space_preferring":
            angle = "brief_and_low_pressure"
            max_points = 1

        if turn_decision.memory_focus == "project_clarity":
            angle = "next_step_clarity"

        return {
            **base,
            "nurture_angle": angle,
            "max_project_points": max_points,
            "engagement_mode": turn_decision.engagement_mode,
        }


_Prev_MemoryWritebackEngine_2 = MemoryWritebackEngine
class MemoryWritebackEngine(_Prev_MemoryWritebackEngine_2):
    def __init__(self, conversation_repo: ConversationRepository, style_profile_repo: UserStyleProfileRepository | None = None, strategy_learning_repo: StrategyLearningRepository | None = None) -> None:
        super().__init__(conversation_repo, style_profile_repo=style_profile_repo, strategy_learning_repo=strategy_learning_repo)
        self.style_profile_repo = style_profile_repo
        self.strategy_learning_repo = strategy_learning_repo

    def write_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        turn_decision: TurnDecision,
        stage_decision: StageDecision,
        understanding: UnderstandingResult,
        final_reply_text: str,
        latest_user_text: str,
    ) -> None:
        super().write_turn(
            business_account_id,
            user_id,
            conversation_id,
            turn_decision,
            stage_decision,
            understanding,
            final_reply_text,
            latest_user_text,
        )
        if self.style_profile_repo:
            existing = self.style_profile_repo.get_profile(business_account_id, user_id) or {}
            notes_json = existing.get("notes_json")
            if isinstance(notes_json, str):
                try:
                    notes_json = json.loads(notes_json)
                except Exception:
                    notes_json = {}
            notes_json = notes_json or {}
            notes_json.update(
                {
                    "memory_focus": turn_decision.memory_focus,
                    "trajectory_goal": turn_decision.trajectory_goal,
                    "relationship_readiness": understanding.relationship_readiness,
                    "relationship_rhythm": turn_decision.relationship_rhythm,
                    "engagement_mode": turn_decision.engagement_mode,
                    "trust_signal": turn_decision.trust_signal,
                }
            )
            self.style_profile_repo.upsert_profile(
                business_account_id,
                user_id,
                turn_decision.user_style_profile,
                turn_decision.tone_bias,
                "slow" if turn_decision.relationship_rhythm in ("slow", "patient") else "normal",
                "ready" if stage_decision.trajectory in ("conversion_window", "project_deepen") else ("cautious" if understanding.hesitation_score > 0.6 else "warming"),
                notes_json=notes_json,
            )
        if self.strategy_learning_repo:
            outcome_signal = "reply_written"
            if understanding.boundary_signal:
                outcome_signal = "space_respected"
            elif stage_decision.trajectory in ("conversion_window", "project_deepen"):
                outcome_signal = "progress_supported"
            self.strategy_learning_repo.insert_record(
                business_account_id,
                user_id,
                conversation_id,
                stage_decision.trajectory,
                turn_decision.user_style_profile,
                turn_decision.reply_goal,
                (
                    f"memory_focus={turn_decision.memory_focus}; tone={turn_decision.tone_bias}; "
                    f"exit={turn_decision.exit_strategy}; rhythm={turn_decision.relationship_rhythm}; "
                    f"engagement={turn_decision.engagement_mode}"
                ),
                outcome_signal,
            )


_Prev_HandoverLearningEngine_1 = HandoverLearningEngine
class HandoverLearningEngine(_Prev_HandoverLearningEngine_1):
    def learn_after_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        context: ConversationContext,
        turn_decision: TurnDecision,
        stage_decision: StageDecision,
        understanding: UnderstandingResult,
    ) -> None:
        if context.manual_takeover_status not in ("pending_resume", "inactive"):
            return
        notes = (
            f"stage={stage_decision.stage}; trajectory={stage_decision.trajectory}; goal={turn_decision.reply_goal}; "
            f"engagement={turn_decision.engagement_mode}; rhythm={turn_decision.relationship_rhythm}; "
            f"trust_signal={turn_decision.trust_signal}; hidden_intent={understanding.hidden_intent}; style={turn_decision.user_style_profile}"
        )
        outcome = "resume_sensitive" if context.manual_takeover_status == "pending_resume" else "turn_completed"
        self.strategy_learning_repo.insert_record(
            business_account_id,
            user_id,
            conversation_id,
            stage_decision.trajectory,
            turn_decision.user_style_profile,
            turn_decision.reply_goal,
            notes,
            outcome,
        )


_Prev_Orchestrator_6 = Orchestrator
class Orchestrator(_Prev_Orchestrator_6):
    def __init__(
        self,
        *args,
        memory_selector: MemorySelector | None = None,
        trajectory_engine: RelationshipTrajectoryEngine | None = None,
        strategy_optimizer: AdaptiveStrategyOptimizer | None = None,
        strategy_learning_repo: StrategyLearningRepository | None = None,
        style_profile_repo: UserStyleProfileRepository | None = None,
        handover_learning_engine: HandoverLearningEngine | None = None,
        rhythm_engine: RelationshipRhythmEngine | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            memory_selector=memory_selector or MemorySelector(),
            trajectory_engine=trajectory_engine or RelationshipTrajectoryEngine(),
            strategy_optimizer=strategy_optimizer or AdaptiveStrategyOptimizer(),
            strategy_learning_repo=strategy_learning_repo,
            style_profile_repo=style_profile_repo,
            handover_learning_engine=handover_learning_engine,
            **kwargs,
        )
        self.rhythm_engine = rhythm_engine or RelationshipRhythmEngine()

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
        try:
            setattr(context, "conversation_id", conversation_id)
        except Exception:
            pass
        user_state = self.user_repo.get_user_state_snapshot(business_account_id, user_id)

        stored_style_profile = self.style_profile_repo.get_profile(business_account_id, user_id) if self.style_profile_repo else None
        recent_strategy_learning = self.strategy_learning_repo.list_recent_for_user(business_account_id, user_id, 6) if self.strategy_learning_repo else []
        try:
            setattr(user_state, "_style_profile_data", stored_style_profile or {})
            setattr(user_state, "_recent_strategy_learning", recent_strategy_learning or [])
        except Exception:
            pass

        ai_allowed, ai_reason = self.ai_switch_engine.decide(
            business_account_id,
            conversation_id,
            user_state.ops_category,
            context.manual_takeover_status,
        )
        if not ai_allowed:
            logger.info("AI reply skipped | conversation_id=%s | reason=%s", conversation_id, ai_reason)
            return

        latest_user_text = (inbound_message.text or "").strip()
        persona_profile = self.persona_profile_builder.build(business_account_id)
        persona_summary = self.persona_core.to_summary() + " " + self.persona_profile_builder.to_summary(persona_profile)
        recent_context = list(context.recent_messages)
        latest_handover_summary = None
        if self.handover_repo and context.manual_takeover_status == "pending_resume":
            latest_handover_summary = self.handover_repo.get_latest_handover_summary_by_conversation(conversation_id)
            if latest_handover_summary:
                recent_context.append(
                    {
                        "sender_type": "system",
                        "message_type": "text",
                        "content_text": f"[ResumeHint] theme={latest_handover_summary.get('theme_summary')}; user_state={latest_handover_summary.get('user_state_summary')}; resume={latest_handover_summary.get('resume_suggestion')}",
                    }
                )

        memory_bundle = self._build_memory_bundle(context, latest_handover_summary)
        latest_summaries = self.conversation_repo.get_recent_summaries(conversation_id, 3)
        understanding = self.understanding_engine.analyze(
            latest_user_text,
            recent_context,
            self.persona_core.to_summary(),
            self._user_state_summary(user_state),
        )
        stage_decision = self.stage_engine.decide(
            understanding,
            context.current_stage,
            user_state=user_state,
            context=context,
            latest_summaries=latest_summaries,
        )
        mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
        turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
        selected_memory = self.memory_selector.select(
            memory_bundle,
            understanding,
            stage_decision.trajectory,
            stored_profile=stored_style_profile,
            recent_learning=recent_strategy_learning,
        )
        style_spec = self.humanization_controller.build_style_spec(
            turn_decision,
            understanding,
            stage_decision,
            user_state,
            persona_profile,
        )
        project_decision = self.project_classifier.classify(
            business_account_id,
            understanding,
            context,
            user_state,
            latest_user_text,
        )
        intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
        segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
        escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
        escalation_decision["should_queue_admin"] = bool(escalation_decision.get("should_queue_admin") or understanding.human_takeover_hint)
        if understanding.human_takeover_hint and not escalation_decision.get("reason"):
            escalation_decision["reason"] = "user asked for a real person or direct explanation"
            escalation_decision["notify_level"] = "suggest_takeover"

        tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
        ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)
        reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
        reply_plan.should_reply = turn_decision.should_reply
        reply_plan.should_continue_product = turn_decision.should_push_project
        reply_plan.should_leave_space = turn_decision.exit_strategy in ("leave_space", "soft_hold")
        reply_plan.should_self_share = turn_decision.self_disclosure_level != "none"

        selected_content = self.content_selector.select(
            business_account_id,
            project_decision.project_id,
            mode_decision.chat_mode,
            reply_plan,
        )
        if hasattr(self, "project_nurture_planner") and self.project_nurture_planner:
            selected_content["v5_project_nurture"] = self.project_nurture_planner.plan(
                understanding,
                stage_decision,
                turn_decision,
                project_decision,
                user_state,
            )
        if not turn_decision.should_push_project and not understanding.explicit_product_query:
            selected_content = {
                k: v for k, v in (selected_content or {}).items()
                if k in ("persona_materials", "self_share_materials", "v5_project_nurture")
            }

        understanding_payload = understanding.__dict__.copy()
        understanding_payload["turn_decision"] = asdict(turn_decision)
        understanding_payload["style_spec"] = asdict(style_spec)
        understanding_payload["memory_summary"] = {
            "recent_messages_count": len(selected_memory.get("selected_recent_messages") or []),
            "has_recent_summary": bool(selected_memory.get("selected_recent_summary")),
            "has_handover_summary": bool(selected_memory.get("selected_handover_summary")),
            "memory_focus": selected_memory.get("memory_focus"),
            "trajectory": stage_decision.trajectory,
            "strategy_context_count": len(selected_memory.get("strategy_context") or []),
            "stored_style_profile": selected_memory.get("stored_style_profile"),
        }
        understanding_payload["relationship_rhythm"] = {
            "rhythm": turn_decision.relationship_rhythm,
            "engagement_mode": turn_decision.engagement_mode,
            "trust_signal": turn_decision.trust_signal,
            "recovery_window_hours": turn_decision.recovery_window_hours,
        }
        if latest_handover_summary:
            understanding_payload["resume_hint"] = latest_handover_summary.get("resume_suggestion")

        draft_reply = self.reply_style_engine.generate(
            latest_user_text,
            selected_memory.get("selected_recent_messages") or recent_context,
            persona_summary,
            self._user_state_summary(user_state),
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding_payload,
            {
                **reply_plan.__dict__,
                "trajectory_goal": turn_decision.trajectory_goal,
                "memory_focus": turn_decision.memory_focus,
                "relationship_rhythm": turn_decision.relationship_rhythm,
                "engagement_mode": turn_decision.engagement_mode,
                "trust_signal": turn_decision.trust_signal,
            },
            selected_content,
        )
        final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        final_reply = FinalReply(
            text=final_text,
            delay_seconds=delay_seconds,
            metadata={
                "turn_decision": asdict(turn_decision),
                "style_spec": asdict(style_spec),
                "trajectory": stage_decision.trajectory,
                "engagement_mode": turn_decision.engagement_mode,
            },
        )

        if reply_plan.should_reply and final_reply.text.strip():
            send_result = self.sender_service.send_text_reply(
                conversation_id,
                final_reply.text,
                final_reply.delay_seconds,
                raw_payload=inbound_message.raw_payload,
                metadata={
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "trajectory": stage_decision.trajectory,
                    "delivery": "queued",
                },
            )
            self.conversation_repo.save_message(
                conversation_id,
                "ai",
                "text",
                final_reply.text,
                {
                    "selected_content": selected_content,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "trajectory": stage_decision.trajectory,
                    "delivery": send_result,
                },
                None,
            )
            self.conversation_repo.set_last_ai_reply_at(conversation_id)

        self.conversation_repo.update_conversation_state(
            conversation_id,
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding.current_mainline_should_continue,
        )
        if project_decision.project_id is not None:
            self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
        self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(
                business_account_id,
                user_id,
                ops_decision["ops_category"],
                ops_decision["reason"],
                "system",
            )
        try:
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "AI Turn Decision",
                {
                    "conversation_id": conversation_id,
                    "stage": stage_decision.stage,
                    "stage_confidence": stage_decision.confidence,
                    "trajectory": stage_decision.trajectory,
                    "chat_mode": mode_decision.chat_mode,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "intent_level": intent_decision.level,
                    "intent_score": intent_decision.score,
                    "stored_style_profile": (stored_style_profile or {}).get("style_profile") if stored_style_profile else None,
                },
            )
        except Exception:
            logger.exception("Failed to create V5 continued AI turn-decision receipt")

        if escalation_decision.get("should_queue_admin"):
            queue_type = "urgent_handover" if escalation_decision.get("notify_level") == "urgent_takeover" else "high_intent"
            priority_score = 95.0 if queue_type == "urgent_handover" else (80.0 if escalation_decision.get("notify_level") == "suggest_takeover" else 60.0)
            self.admin_queue_repo.upsert_queue_item(
                business_account_id,
                user_id,
                queue_type,
                priority_score,
                escalation_decision["reason"],
            )
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "High intent detected",
                {
                    "notify_level": escalation_decision.get("notify_level"),
                    "reason": escalation_decision.get("reason"),
                    "project_id": project_decision.project_id,
                    "segment_name": segment_decision.get("segment_name"),
                    "intent_level": intent_decision.level,
                    "intent_score": intent_decision.score,
                    "style_spec": asdict(style_spec),
                    "trajectory": stage_decision.trajectory,
                    "engagement_mode": turn_decision.engagement_mode,
                },
            )
            if self.admin_notifier:
                self.admin_notifier.notify_high_intent(
                    business_account_id,
                    user_id,
                    conversation_id,
                    escalation_decision.get("notify_level") or "watch",
                    escalation_decision.get("reason") or "",
                )

        if hasattr(self, "memory_writeback_engine") and self.memory_writeback_engine and turn_decision.should_write_memory:
            try:
                self.memory_writeback_engine.write_turn(
                    business_account_id,
                    user_id,
                    conversation_id,
                    turn_decision,
                    stage_decision,
                    understanding,
                    final_reply.text,
                    latest_user_text,
                )
            except Exception:
                logger.exception("V5 continued memory writeback failed")

        if self.handover_learning_engine:
            try:
                self.handover_learning_engine.learn_after_turn(
                    business_account_id,
                    user_id,
                    conversation_id,
                    context,
                    turn_decision,
                    stage_decision,
                    understanding,
                )
            except Exception:
                logger.exception("V5 continued handover learning failed")

        if hasattr(self, "followup_scheduler") and self.followup_scheduler and turn_decision.need_followup:
            try:
                self.followup_scheduler.schedule(
                    business_account_id,
                    user_id,
                    conversation_id,
                    turn_decision,
                    understanding,
                    stage_decision,
                )
            except Exception:
                logger.exception("V5 continued followup scheduling failed")


# =========================
# V5.1 continued optimization
# =========================

def ensure_v51_upgrade_schema(db: Database) -> None:
    with db.transaction():
        with db.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_maturity_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    business_account_id BIGINT NOT NULL,
                    user_id BIGINT NOT NULL,
                    conversation_id BIGINT NOT NULL,
                    maturity_label TEXT,
                    maturity_score DOUBLE PRECISION DEFAULT 0.0,
                    relationship_rhythm TEXT,
                    engagement_mode TEXT,
                    trust_signal TEXT,
                    memory_priority TEXT,
                    continuity_mode TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_maturity_snapshots_user_created
                ON conversation_maturity_snapshots (business_account_id, user_id, created_at DESC)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS manual_strategy_abstractions (
                    id BIGSERIAL PRIMARY KEY,
                    business_account_id BIGINT NOT NULL,
                    user_id BIGINT NOT NULL,
                    conversation_id BIGINT NOT NULL,
                    abstraction_type TEXT,
                    abstraction_text TEXT,
                    source_signal TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_manual_strategy_abs_user_created
                ON manual_strategy_abstractions (business_account_id, user_id, created_at DESC)
                """
            )


class MaturitySnapshotRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_snapshot(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        maturity_label: str,
        maturity_score: float,
        relationship_rhythm: str,
        engagement_mode: str,
        trust_signal: str,
        memory_priority: str,
        continuity_mode: str,
    ) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_maturity_snapshots (
                        business_account_id, user_id, conversation_id, maturity_label,
                        maturity_score, relationship_rhythm, engagement_mode, trust_signal,
                        memory_priority, continuity_mode
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        business_account_id,
                        user_id,
                        conversation_id,
                        maturity_label,
                        float(maturity_score or 0.0),
                        relationship_rhythm,
                        engagement_mode,
                        trust_signal,
                        memory_priority,
                        continuity_mode,
                    ),
                )

    def list_recent_for_user(self, business_account_id: int, user_id: int, limit: int = 6) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM conversation_maturity_snapshots
                WHERE business_account_id=%s AND user_id=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (business_account_id, user_id, limit),
            )
            return list(cur.fetchall() or [])


class ManualStrategyAbstractionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_abstraction(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        abstraction_type: str,
        abstraction_text: str,
        source_signal: str,
    ) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO manual_strategy_abstractions (
                        business_account_id, user_id, conversation_id,
                        abstraction_type, abstraction_text, source_signal
                    ) VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        business_account_id,
                        user_id,
                        conversation_id,
                        abstraction_type,
                        (abstraction_text or "")[:3000],
                        source_signal,
                    ),
                )

    def list_recent_for_user(self, business_account_id: int, user_id: int, limit: int = 8) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM manual_strategy_abstractions
                WHERE business_account_id=%s AND user_id=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (business_account_id, user_id, limit),
            )
            return list(cur.fetchall() or [])


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
    emotion_strength: float = 0.4
    boundary_strength: float = 0.0
    social_openness: float = 0.45
    project_relevance: float = 0.0
    human_takeover_hint: bool = False
    notes: list[str] = field(default_factory=list)
    hidden_intent: str | None = None
    conversation_energy: str = "steady"
    hesitation_score: float = 0.0
    style_signal: str = "balanced"
    relationship_readiness: str = "warmup"
    reassurance_need: float = 0.0
    continuity_need: float = 0.0
    pressure_risk: float = 0.0


@dataclass
class StageDecision:
    stage: str
    changed: bool
    reason: str
    confidence: float = 0.65
    trajectory: str = "steady"
    momentum: str = "neutral"
    maturity_state: str = "emerging"
    maturity_reason: str = ""
    maturity_score: float = 0.5


@dataclass
class TurnDecision:
    reply_goal: str = "rapport"
    should_push_project: bool = False
    social_distance: str = "balanced"
    reply_length: str = "medium"
    ask_followup_question: bool = False
    exit_strategy: str = "gentle_close"
    self_disclosure_level: str = "light"
    reason: str = ""
    should_reply: bool = True
    marketing_intensity: str = "subtle"
    tone_bias: str = "natural"
    need_followup: bool = False
    followup_type: str | None = None
    followup_delay_seconds: int | None = None
    nurture_goal: str | None = None
    should_write_memory: bool = True
    trajectory_goal: str | None = None
    memory_focus: str | None = None
    user_style_profile: str = "balanced"
    project_strategy: str | None = None
    relationship_rhythm: str = "steady"
    engagement_mode: str = "maintain"
    recovery_window_hours: int = 24
    trust_signal: str = "neutral"
    maturity_target: str = "stable_presence"
    memory_priority: str = "relationship"
    continuity_mode: str = "implicit"
    project_window: str = "hold"
    maturity_score: float = 0.5


@dataclass
class StyleSpec:
    tone: str = "natural"
    formality: str = "balanced"
    warmth: str = "medium"
    length: str = "medium"
    question_rate: str = "low"
    emoji_usage: str = "minimal"
    self_disclosure_ratio: str = "none"
    completion_level: str = "balanced"
    marketing_visibility: str = "subtle"
    naturalness_bias: str = "high"
    cadence_bias: str = "steady"
    initiative_level: str = "low"
    wording_texture: str = "smooth"
    pause_texture: str = "natural"
    anchor_style: str = "light"
    maturity_polish: str = "balanced"
    directness: str = "balanced"


class RelationshipMaturityEngine:
    def assess(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        stored_profile: dict[str, Any] | None = None,
        recent_learning: list[dict] | None = None,
    ) -> dict[str, Any]:
        stored_profile = stored_profile or {}
        recent_learning = recent_learning or []
        score = 0.46
        label = "emerging"
        reasons: list[str] = []
        if stage_decision.trajectory in ("rapport_building", "project_deepen"):
            score += 0.12
            reasons.append("trajectory_progress")
        if stage_decision.trajectory == "conversion_window":
            score += 0.18
            reasons.append("conversion_window")
        if understanding.boundary_signal or user_state.boundary_sensitivity > 0.72:
            score -= 0.16
            reasons.append("boundary_pressure")
        if understanding.hesitation_score > 0.62:
            score -= 0.08
            reasons.append("hesitation")
        if understanding.social_openness > 0.62 and user_state.trust_score > 0.58:
            score += 0.08
            reasons.append("trust_and_openness")
        if (stored_profile.get("project_readiness") or "").lower() == "ready":
            score += 0.06
            reasons.append("stored_ready")
        if any((r.get("outcome_signal") or "") in ("space_respected", "progress_supported") for r in recent_learning[:3]):
            score += 0.04
            reasons.append("recent_good_outcomes")
        score = max(0.08, min(0.96, score))
        if score >= 0.8:
            label = "mature_progress"
        elif score >= 0.66:
            label = "stable_building"
        elif score >= 0.52:
            label = "careful_growth"
        elif understanding.boundary_signal or stage_decision.trajectory in ("cooldown", "pause_and_return"):
            label = "fragile_space"
        reason = ",".join(reasons) if reasons else "default_maturity"
        return {"maturity_state": label, "maturity_reason": reason, "maturity_score": score}


class MemoryPriorityResolver:
    def resolve(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        turn_preview: dict[str, Any] | None = None,
        stored_profile: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        stored_profile = stored_profile or {}
        priority = "relationship"
        continuity_mode = "implicit"
        if understanding.boundary_signal or stage_decision.trajectory in ("cooldown", "pause_and_return"):
            priority = "boundary"
            continuity_mode = "light_touch"
        elif understanding.reassurance_need > 0.64 or understanding.hesitation_score > 0.62:
            priority = "reassurance"
            continuity_mode = "gentle_reference"
        elif stage_decision.trajectory in ("conversion_window", "project_deepen") or understanding.explicit_product_query:
            priority = "project_clarity"
            continuity_mode = "fact_link"
        elif understanding.continuity_need > 0.58 or (stored_profile.get("style_profile") or "") == "rapport_first":
            priority = "relationship"
            continuity_mode = "soft_reference"
        return {"memory_priority": priority, "continuity_mode": continuity_mode}


class ContinuityGuard:
    def trim_recent(self, selected_recent: list[dict], continuity_mode: str, priority: str) -> list[dict]:
        msgs = list(selected_recent or [])
        if continuity_mode == "light_touch":
            return msgs[-4:]
        if continuity_mode == "fact_link" or priority == "project_clarity":
            return msgs[-8:]
        if continuity_mode == "soft_reference":
            return msgs[-6:]
        return msgs[-5:]

    def should_surface_summary(self, understanding: UnderstandingResult, priority: str) -> bool:
        if understanding.boundary_signal and priority == "boundary":
            return False
        if understanding.continuity_need > 0.5:
            return True
        return priority in ("project_clarity", "reassurance")


class StyleStabilityScorer:
    def stabilize(self, proposed_style: str, stored_profile: dict[str, Any] | None = None, recent_learning: list[dict] | None = None) -> str:
        stored_profile = stored_profile or {}
        recent_learning = recent_learning or []
        stored_style = (stored_profile.get("style_profile") or "").lower()
        recent_styles = [str(r.get("user_style_profile") or "").lower() for r in recent_learning[:5] if r.get("user_style_profile")]
        if stored_style and proposed_style != stored_style:
            if recent_styles.count(proposed_style) < 2 and proposed_style != "balanced":
                return stored_style
        if proposed_style == "balanced" and stored_style:
            return stored_style
        return proposed_style


class ProjectWindowEvaluator:
    def evaluate(self, understanding: UnderstandingResult, stage_decision: StageDecision, turn_decision: TurnDecision) -> str:
        if understanding.boundary_signal or turn_decision.engagement_mode == "protect_space":
            return "hold"
        if stage_decision.trajectory == "conversion_window" and understanding.pressure_risk < 0.55:
            return "open"
        if stage_decision.trajectory in ("project_deepen",) or turn_decision.engagement_mode == "clarify_and_progress":
            return "careful_open"
        if understanding.reassurance_need > 0.6 or turn_decision.engagement_mode == "reassure_and_hold":
            return "soft_background"
        return "light_only"


class ManualStrategyAbstractor:
    def abstract(self, context: ConversationContext, turn_decision: TurnDecision, stage_decision: StageDecision, understanding: UnderstandingResult) -> dict[str, str] | None:
        if context.manual_takeover_status not in ("pending_resume", "inactive"):
            return None
        abstraction_type = "resume_bridge"
        if turn_decision.engagement_mode == "protect_space":
            abstraction_type = "boundary_protection"
        elif turn_decision.engagement_mode == "clarify_and_progress":
            abstraction_type = "low_pressure_progress"
        elif turn_decision.engagement_mode == "reassure_and_hold":
            abstraction_type = "reassurance_hold"
        text = (
            f"mode={turn_decision.engagement_mode}; rhythm={turn_decision.relationship_rhythm}; "
            f"maturity={turn_decision.maturity_target}; focus={turn_decision.memory_priority}; "
            f"window={turn_decision.project_window}; hidden={understanding.hidden_intent or 'none'}"
        )
        return {"abstraction_type": abstraction_type, "abstraction_text": text, "source_signal": stage_decision.trajectory}


class ResumeBridgePlanner:
    def build(self, turn_decision: TurnDecision, understanding: UnderstandingResult) -> str:
        if turn_decision.engagement_mode == "protect_space":
            return "resume_with_acknowledgment_only"
        if understanding.reassurance_need > 0.58:
            return "resume_with_gentle_reassurance"
        if turn_decision.project_window in ("open", "careful_open"):
            return "resume_with_light_clarity"
        return "resume_with_soft_presence"


class MaturityPolishEngine:
    def polish(self, text: str, understanding: UnderstandingResult, style_spec: StyleSpec, turn_decision: TurnDecision) -> str:
        t = " ".join((text or "").split()).strip()
        if not t:
            return t
        low_maturity_fillers = [
            "just to be honest",
            "to be completely honest",
            "please feel free",
            "at the end of the day",
            "definitely",
            "absolutely",
        ]
        lowered = t.lower()
        for frag in low_maturity_fillers:
            if frag in lowered:
                t = re.sub(re.escape(frag), "", t, flags=re.IGNORECASE)
                lowered = t.lower()
        if style_spec.maturity_polish == "restrained":
            t = t.replace("!", ".")
        if turn_decision.engagement_mode == "protect_space":
            t = re.sub(r'\b(can i|would you like|shall i|do you want to)\b', '', t, flags=re.IGNORECASE)
            t = t.replace("??", "?")
            if t.endswith("?"):
                t = t[:-1].rstrip() + "."
        if understanding.pressure_risk > 0.64:
            for frag in ["right now", "today", "next step", "we can move forward"]:
                t = re.sub(re.escape(frag), "", t, flags=re.IGNORECASE)
        t = re.sub(r'\s{2,}', ' ', t).strip(" ,.-")
        if t and not t.endswith((".", "!", "?")):
            t += "."
        return t


_Prev_UserUnderstandingEngine_4 = UserUnderstandingEngine
class UserUnderstandingEngine(_Prev_UserUnderstandingEngine_4):
    def analyze(self, latest_user_text: str, recent_context: list[dict], persona_summary: str, user_state_summary: str) -> UnderstandingResult:
        result = super().analyze(latest_user_text, recent_context, persona_summary, user_state_summary)
        reassurance_need = 0.15
        continuity_need = 0.25
        pressure_risk = 0.18
        text = (latest_user_text or "").lower()
        if result.boundary_signal:
            reassurance_need += 0.18
            continuity_need = min(0.45, continuity_need)
            pressure_risk += 0.28
        if result.hesitation_score > 0.62 or result.hidden_intent in ("curious_but_cautious", "low_pressure_probe"):
            reassurance_need += 0.24
            pressure_risk += 0.14
        if any(w in text for w in ["again", "as i said", "last time", "before", "earlier"]):
            continuity_need += 0.34
        if result.explicit_product_query:
            continuity_need += 0.16
        if result.high_intent_signal:
            pressure_risk = max(0.12, pressure_risk - 0.08)
        result.reassurance_need = max(0.0, min(1.0, reassurance_need))
        result.continuity_need = max(0.0, min(1.0, continuity_need))
        result.pressure_risk = max(0.0, min(1.0, pressure_risk))
        if result.reassurance_need > 0.58 and "reassurance_needed" not in result.notes:
            result.notes.append("reassurance_needed")
        if result.continuity_need > 0.58 and "continuity_needed" not in result.notes:
            result.notes.append("continuity_needed")
        return result


_Prev_ConversationStageEngine_5 = ConversationStageEngine
class ConversationStageEngine(_Prev_ConversationStageEngine_5):
    def __init__(self, trajectory_engine: RelationshipTrajectoryEngine | None = None, maturity_engine: RelationshipMaturityEngine | None = None) -> None:
        super().__init__(trajectory_engine=trajectory_engine or RelationshipTrajectoryEngine())
        self.maturity_engine = maturity_engine or RelationshipMaturityEngine()

    def decide(
        self,
        understanding: UnderstandingResult,
        current_stage: str | None,
        user_state: UserStateSnapshot | None = None,
        context: ConversationContext | None = None,
        latest_summaries: list[dict] | None = None,
    ) -> StageDecision:
        stage = super().decide(understanding, current_stage, user_state=user_state, context=context, latest_summaries=latest_summaries)
        profile = getattr(user_state, "_style_profile_data", None) or {}
        learning = getattr(user_state, "_recent_strategy_learning", None) or []
        maturity = self.maturity_engine.assess(understanding, stage, user_state or UserStateSnapshot(), stored_profile=profile, recent_learning=learning)
        return StageDecision(
            stage=stage.stage,
            changed=stage.changed,
            reason=stage.reason,
            confidence=stage.confidence,
            trajectory=stage.trajectory,
            momentum=stage.momentum,
            maturity_state=maturity["maturity_state"],
            maturity_reason=maturity["maturity_reason"],
            maturity_score=maturity["maturity_score"],
        )


_Prev_AdaptiveStrategyOptimizer_2 = AdaptiveStrategyOptimizer
class AdaptiveStrategyOptimizer(_Prev_AdaptiveStrategyOptimizer_2):
    def __init__(self) -> None:
        super().__init__()
        self.stability_scorer = StyleStabilityScorer()

    def infer_user_style(
        self,
        understanding: UnderstandingResult,
        user_state: UserStateSnapshot,
        stored_profile: dict[str, Any] | None = None,
        recent_learning: list[dict] | None = None,
    ) -> str:
        proposed = super().infer_user_style(
            understanding,
            user_state,
            stored_profile=stored_profile,
            recent_learning=recent_learning,
        )
        return self.stability_scorer.stabilize(proposed, stored_profile=stored_profile, recent_learning=recent_learning)


_Prev_MemorySelector_2 = MemorySelector
class MemorySelector(_Prev_MemorySelector_2):
    def __init__(self, resolver: MemoryPriorityResolver | None = None, guard: ContinuityGuard | None = None) -> None:
        super().__init__()
        self.resolver = resolver or MemoryPriorityResolver()
        self.guard = guard or ContinuityGuard()

    def select(
        self,
        memory_bundle: MemoryBundle,
        understanding: UnderstandingResult,
        trajectory: str,
        stored_profile: dict[str, Any] | None = None,
        recent_learning: list[dict] | None = None,
        stage_decision: StageDecision | None = None,
    ) -> dict[str, Any]:
        base = super().select(
            memory_bundle,
            understanding,
            trajectory,
            stored_profile=stored_profile,
            recent_learning=recent_learning,
        )
        stage_decision = stage_decision or StageDecision(stage="daily_rapport", changed=False, reason="selector_default", trajectory=trajectory)
        resolved = self.resolver.resolve(understanding, stage_decision, stored_profile=stored_profile)
        selected_recent = self.guard.trim_recent(base.get("selected_recent_messages") or [], resolved["continuity_mode"], resolved["memory_priority"])
        selected_summary = base.get("selected_recent_summary") if self.guard.should_surface_summary(understanding, resolved["memory_priority"]) else ""
        strategy_context = []
        for item in (recent_learning or [])[:4]:
            snippet = (item.get("strategy_notes") or "")[:220]
            if snippet:
                strategy_context.append(snippet)
        return {
            **base,
            "selected_recent_messages": selected_recent,
            "selected_recent_summary": selected_summary,
            "memory_focus": resolved["memory_priority"],
            "memory_priority": resolved["memory_priority"],
            "continuity_mode": resolved["continuity_mode"],
            "strategy_context": strategy_context,
        }


_Prev_TurnDecisionEngine_6 = TurnDecisionEngine
class TurnDecisionEngine(_Prev_TurnDecisionEngine_6):
    def __init__(
        self,
        optimizer: AdaptiveStrategyOptimizer | None = None,
        rhythm_engine: RelationshipRhythmEngine | None = None,
        maturity_engine: RelationshipMaturityEngine | None = None,
        project_window_evaluator: ProjectWindowEvaluator | None = None,
        memory_resolver: MemoryPriorityResolver | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer or AdaptiveStrategyOptimizer(), rhythm_engine=rhythm_engine or RelationshipRhythmEngine())
        self.optimizer = optimizer or AdaptiveStrategyOptimizer()
        self.rhythm_engine = rhythm_engine or RelationshipRhythmEngine()
        self.maturity_engine = maturity_engine or RelationshipMaturityEngine()
        self.project_window_evaluator = project_window_evaluator or ProjectWindowEvaluator()
        self.memory_resolver = memory_resolver or MemoryPriorityResolver()

    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        decision = super().decide(understanding, stage_decision, mode_decision, user_state)
        stored_profile = getattr(user_state, "_style_profile_data", None) or {}
        maturity = self.maturity_engine.assess(understanding, stage_decision, user_state, stored_profile=stored_profile, recent_learning=getattr(user_state, "_recent_strategy_learning", None) or [])
        resolved_memory = self.memory_resolver.resolve(understanding, stage_decision, stored_profile=stored_profile)
        project_window = self.project_window_evaluator.evaluate(understanding, stage_decision, decision)
        data = {**asdict(decision)}
        data["maturity_target"] = maturity["maturity_state"]
        data["maturity_score"] = maturity["maturity_score"]
        data["memory_priority"] = resolved_memory["memory_priority"]
        data["continuity_mode"] = resolved_memory["continuity_mode"]
        data["project_window"] = project_window
        if project_window == "hold":
            data["should_push_project"] = False
            data["marketing_intensity"] = "none"
        elif project_window == "soft_background":
            data["marketing_intensity"] = "subtle"
            if data.get("reply_length") == "medium":
                data["reply_length"] = "short"
        if maturity["maturity_state"] == "fragile_space":
            data["ask_followup_question"] = False
            data["exit_strategy"] = "leave_space"
            data["social_distance"] = "give_space"
        if understanding.reassurance_need > 0.62 and data.get("reply_goal") not in ("respect_boundary",):
            data["reply_goal"] = "reassure_before_progress"
            data["marketing_intensity"] = "none" if project_window in ("hold", "soft_background") else data.get("marketing_intensity")
        data["reason"] = (data.get("reason") or "") + f"; maturity={maturity['maturity_state']}; mem={resolved_memory['memory_priority']}; window={project_window}"
        return TurnDecision(**data)


_Prev_HumanizationController_5 = HumanizationController
class HumanizationController(_Prev_HumanizationController_5):
    def __init__(self, optimizer: AdaptiveStrategyOptimizer | None = None) -> None:
        super().__init__(optimizer=optimizer or AdaptiveStrategyOptimizer())

    def build_style_spec(
        self,
        decision: TurnDecision,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        persona_profile: dict[str, Any] | None = None,
    ) -> StyleSpec:
        base = super().build_style_spec(decision, understanding, stage_decision, user_state, persona_profile)
        maturity_polish = "balanced"
        directness = "balanced"
        if decision.maturity_target in ("fragile_space",):
            maturity_polish = "restrained"
            directness = "low"
        elif decision.user_style_profile == "logic_first":
            maturity_polish = "clean"
            directness = "high"
        elif decision.user_style_profile == "emotion_first":
            maturity_polish = "warm"
            directness = "soft"
        elif understanding.reassurance_need > 0.58:
            maturity_polish = "calm"
            directness = "soft"
        return StyleSpec(
            tone=base.tone,
            formality=base.formality,
            warmth=base.warmth,
            length=base.length,
            question_rate=base.question_rate,
            emoji_usage=base.emoji_usage,
            self_disclosure_ratio=base.self_disclosure_ratio,
            completion_level=base.completion_level,
            marketing_visibility=base.marketing_visibility,
            naturalness_bias=base.naturalness_bias,
            cadence_bias=base.cadence_bias,
            initiative_level=base.initiative_level,
            wording_texture=base.wording_texture,
            pause_texture=base.pause_texture,
            anchor_style=base.anchor_style,
            maturity_polish=maturity_polish,
            directness=directness,
        )


_Prev_ProjectNurturePlanner_3 = ProjectNurturePlanner
class ProjectNurturePlanner(_Prev_ProjectNurturePlanner_3):
    def __init__(self, window_evaluator: ProjectWindowEvaluator | None = None) -> None:
        super().__init__()
        self.window_evaluator = window_evaluator or ProjectWindowEvaluator()

    def plan(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        turn_decision: TurnDecision,
        project_decision: ProjectDecision,
        user_state: UserStateSnapshot,
    ) -> dict[str, Any]:
        base = super().plan(understanding, stage_decision, turn_decision, project_decision, user_state)
        if not base.get("should_include_project_content"):
            return base
        window = self.window_evaluator.evaluate(understanding, stage_decision, turn_decision)
        angle = base.get("nurture_angle") or turn_decision.project_strategy or "single_clear_point"
        if window == "hold":
            return {**base, "should_include_project_content": False, "project_window": window}
        if window == "soft_background":
            angle = "reassure_then_light_value"
        elif window == "careful_open" and angle == "single_clear_point":
            angle = "clear_fact_then_soft_bridge"
        elif window == "open":
            angle = "next_step_clarity"
        return {**base, "nurture_angle": angle, "project_window": window}


_Prev_MemoryWritebackEngine_3 = MemoryWritebackEngine
class MemoryWritebackEngine(_Prev_MemoryWritebackEngine_3):
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        style_profile_repo: UserStyleProfileRepository | None = None,
        strategy_learning_repo: StrategyLearningRepository | None = None,
        maturity_snapshot_repo: MaturitySnapshotRepository | None = None,
    ) -> None:
        super().__init__(conversation_repo, style_profile_repo=style_profile_repo, strategy_learning_repo=strategy_learning_repo)
        self.maturity_snapshot_repo = maturity_snapshot_repo

    def write_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        turn_decision: TurnDecision,
        stage_decision: StageDecision,
        understanding: UnderstandingResult,
        final_reply_text: str,
        latest_user_text: str,
    ) -> None:
        super().write_turn(business_account_id, user_id, conversation_id, turn_decision, stage_decision, understanding, final_reply_text, latest_user_text)
        if self.maturity_snapshot_repo:
            self.maturity_snapshot_repo.insert_snapshot(
                business_account_id,
                user_id,
                conversation_id,
                turn_decision.maturity_target,
                turn_decision.maturity_score,
                turn_decision.relationship_rhythm,
                turn_decision.engagement_mode,
                turn_decision.trust_signal,
                turn_decision.memory_priority,
                turn_decision.continuity_mode,
            )


_Prev_HandoverLearningEngine_2 = HandoverLearningEngine
class HandoverLearningEngine(_Prev_HandoverLearningEngine_2):
    def __init__(
        self,
        strategy_learning_repo: StrategyLearningRepository,
        abstraction_repo: ManualStrategyAbstractionRepository | None = None,
        abstractor: ManualStrategyAbstractor | None = None,
        resume_bridge_planner: ResumeBridgePlanner | None = None,
    ) -> None:
        super().__init__(strategy_learning_repo)
        self.abstraction_repo = abstraction_repo
        self.abstractor = abstractor or ManualStrategyAbstractor()
        self.resume_bridge_planner = resume_bridge_planner or ResumeBridgePlanner()

    def learn_after_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        context: ConversationContext,
        turn_decision: TurnDecision,
        stage_decision: StageDecision,
        understanding: UnderstandingResult,
    ) -> None:
        super().learn_after_turn(business_account_id, user_id, conversation_id, context, turn_decision, stage_decision, understanding)
        abstraction = self.abstractor.abstract(context, turn_decision, stage_decision, understanding)
        if self.abstraction_repo and abstraction:
            extra = abstraction["abstraction_text"] + f"; resume_bridge={self.resume_bridge_planner.build(turn_decision, understanding)}"
            self.abstraction_repo.insert_abstraction(
                business_account_id,
                user_id,
                conversation_id,
                abstraction["abstraction_type"],
                extra,
                abstraction["source_signal"],
            )


_Prev_ReplySelfCheckEngine_4 = ReplySelfCheckEngine
class ReplySelfCheckEngine(_Prev_ReplySelfCheckEngine_4):
    def __init__(self, polish_engine: MaturityPolishEngine | None = None) -> None:
        super().__init__()
        self.polish_engine = polish_engine or MaturityPolishEngine()

    def check_and_fix_with_context(
        self,
        draft_reply: str,
        mode: str,
        understanding: UnderstandingResult,
        style_spec: StyleSpec,
        turn_decision: TurnDecision,
    ) -> str:
        text = super().check_and_fix(draft_reply, mode, understanding)
        text = self.polish_engine.polish(text, understanding, style_spec, turn_decision)
        if style_spec.directness == "high" and len(text) > 280:
            text = text[:280].rstrip(" ,")
            if not text.endswith((".", "!", "?")):
                text += "."
        if style_spec.maturity_polish == "warm" and understanding.emotion_state in ("low", "anxious") and not text.lower().startswith(("i get", "that sounds", "no worries")):
            text = "I get that. " + text[0].lower() + text[1:] if len(text) > 1 else text
        return " ".join(text.split()).strip()


_Prev_Orchestrator_7 = Orchestrator
class Orchestrator(_Prev_Orchestrator_7):
    def __init__(
        self,
        *args,
        maturity_snapshot_repo: MaturitySnapshotRepository | None = None,
        manual_strategy_repo: ManualStrategyAbstractionRepository | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.maturity_snapshot_repo = maturity_snapshot_repo
        self.manual_strategy_repo = manual_strategy_repo

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
        try:
            setattr(context, "conversation_id", conversation_id)
        except Exception:
            pass
        user_state = self.user_repo.get_user_state_snapshot(business_account_id, user_id)

        stored_style_profile = self.style_profile_repo.get_profile(business_account_id, user_id) if self.style_profile_repo else None
        recent_strategy_learning = self.strategy_learning_repo.list_recent_for_user(business_account_id, user_id, 6) if self.strategy_learning_repo else []
        recent_manual_abstractions = self.manual_strategy_repo.list_recent_for_user(business_account_id, user_id, 4) if self.manual_strategy_repo else []
        recent_maturity = self.maturity_snapshot_repo.list_recent_for_user(business_account_id, user_id, 3) if self.maturity_snapshot_repo else []
        try:
            setattr(user_state, "_style_profile_data", stored_style_profile or {})
            setattr(user_state, "_recent_strategy_learning", recent_strategy_learning or [])
            setattr(user_state, "_recent_manual_abstractions", recent_manual_abstractions or [])
            setattr(user_state, "_recent_maturity_snapshots", recent_maturity or [])
        except Exception:
            pass

        ai_allowed, ai_reason = self.ai_switch_engine.decide(
            business_account_id,
            conversation_id,
            user_state.ops_category,
            context.manual_takeover_status,
        )
        if not ai_allowed:
            logger.info("AI reply skipped | conversation_id=%s | reason=%s", conversation_id, ai_reason)
            return

        latest_user_text = (inbound_message.text or "").strip()
        persona_profile = self.persona_profile_builder.build(business_account_id)
        persona_summary = self.persona_core.to_summary() + " " + self.persona_profile_builder.to_summary(persona_profile)
        recent_context = list(context.recent_messages)
        latest_handover_summary = None
        if self.handover_repo and context.manual_takeover_status == "pending_resume":
            latest_handover_summary = self.handover_repo.get_latest_handover_summary_by_conversation(conversation_id)
            if latest_handover_summary:
                recent_context.append(
                    {
                        "sender_type": "system",
                        "message_type": "text",
                        "content_text": f"[ResumeHint] theme={latest_handover_summary.get('theme_summary')}; user_state={latest_handover_summary.get('user_state_summary')}; resume={latest_handover_summary.get('resume_suggestion')}",
                    }
                )

        memory_bundle = self._build_memory_bundle(context, latest_handover_summary)
        latest_summaries = self.conversation_repo.get_recent_summaries(conversation_id, 3)
        understanding = self.understanding_engine.analyze(
            latest_user_text,
            recent_context,
            self.persona_core.to_summary(),
            self._user_state_summary(user_state),
        )
        stage_decision = self.stage_engine.decide(
            understanding,
            context.current_stage,
            user_state=user_state,
            context=context,
            latest_summaries=latest_summaries,
        )
        mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
        turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
        selected_memory = self.memory_selector.select(
            memory_bundle,
            understanding,
            stage_decision.trajectory,
            stored_profile=stored_style_profile,
            recent_learning=recent_strategy_learning,
            stage_decision=stage_decision,
        )
        style_spec = self.humanization_controller.build_style_spec(
            turn_decision,
            understanding,
            stage_decision,
            user_state,
            persona_profile,
        )
        project_decision = self.project_classifier.classify(
            business_account_id,
            understanding,
            context,
            user_state,
            latest_user_text,
        )
        intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
        segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
        escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
        escalation_decision["should_queue_admin"] = bool(escalation_decision.get("should_queue_admin") or understanding.human_takeover_hint)
        if understanding.human_takeover_hint and not escalation_decision.get("reason"):
            escalation_decision["reason"] = "user asked for a real person or direct explanation"
            escalation_decision["notify_level"] = "suggest_takeover"

        tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
        ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)
        reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
        reply_plan.should_reply = turn_decision.should_reply
        reply_plan.should_continue_product = turn_decision.should_push_project
        reply_plan.should_leave_space = turn_decision.exit_strategy in ("leave_space", "soft_hold")
        reply_plan.should_self_share = turn_decision.self_disclosure_level != "none"

        selected_content = self.content_selector.select(
            business_account_id,
            project_decision.project_id,
            mode_decision.chat_mode,
            reply_plan,
        )
        if hasattr(self, "project_nurture_planner") and self.project_nurture_planner:
            selected_content["v5_project_nurture"] = self.project_nurture_planner.plan(
                understanding,
                stage_decision,
                turn_decision,
                project_decision,
                user_state,
            )
        if not turn_decision.should_push_project and not understanding.explicit_product_query:
            selected_content = {
                k: v for k, v in (selected_content or {}).items()
                if k in ("persona_materials", "self_share_materials", "v5_project_nurture")
            }

        understanding_payload = understanding.__dict__.copy()
        understanding_payload["turn_decision"] = asdict(turn_decision)
        understanding_payload["style_spec"] = asdict(style_spec)
        understanding_payload["memory_summary"] = {
            "recent_messages_count": len(selected_memory.get("selected_recent_messages") or []),
            "has_recent_summary": bool(selected_memory.get("selected_recent_summary")),
            "has_handover_summary": bool(selected_memory.get("selected_handover_summary")),
            "memory_focus": selected_memory.get("memory_focus"),
            "memory_priority": selected_memory.get("memory_priority"),
            "continuity_mode": selected_memory.get("continuity_mode"),
            "trajectory": stage_decision.trajectory,
            "strategy_context_count": len(selected_memory.get("strategy_context") or []),
            "stored_style_profile": selected_memory.get("stored_style_profile"),
        }
        understanding_payload["relationship_rhythm"] = {
            "rhythm": turn_decision.relationship_rhythm,
            "engagement_mode": turn_decision.engagement_mode,
            "trust_signal": turn_decision.trust_signal,
            "recovery_window_hours": turn_decision.recovery_window_hours,
            "maturity_target": turn_decision.maturity_target,
            "project_window": turn_decision.project_window,
        }
        understanding_payload["manual_strategy_context"] = [
            (item.get("abstraction_text") or "")[:220] for item in (recent_manual_abstractions or []) if item.get("abstraction_text")
        ]
        if latest_handover_summary:
            understanding_payload["resume_hint"] = latest_handover_summary.get("resume_suggestion")

        draft_reply = self.reply_style_engine.generate(
            latest_user_text,
            selected_memory.get("selected_recent_messages") or recent_context,
            persona_summary,
            self._user_state_summary(user_state),
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding_payload,
            {
                **reply_plan.__dict__,
                "trajectory_goal": turn_decision.trajectory_goal,
                "memory_focus": turn_decision.memory_focus,
                "relationship_rhythm": turn_decision.relationship_rhythm,
                "engagement_mode": turn_decision.engagement_mode,
                "trust_signal": turn_decision.trust_signal,
                "maturity_target": turn_decision.maturity_target,
                "memory_priority": turn_decision.memory_priority,
                "continuity_mode": turn_decision.continuity_mode,
                "project_window": turn_decision.project_window,
            },
            selected_content,
        )
        if hasattr(self.reply_self_check_engine, "check_and_fix_with_context"):
            final_text = self.reply_self_check_engine.check_and_fix_with_context(
                draft_reply,
                mode_decision.chat_mode,
                understanding,
                style_spec,
                turn_decision,
            )
        else:
            final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        final_reply = FinalReply(
            text=final_text,
            delay_seconds=delay_seconds,
            metadata={
                "turn_decision": asdict(turn_decision),
                "style_spec": asdict(style_spec),
                "trajectory": stage_decision.trajectory,
                "engagement_mode": turn_decision.engagement_mode,
                "maturity_target": turn_decision.maturity_target,
            },
        )

        if reply_plan.should_reply and final_reply.text.strip():
            send_result = self.sender_service.send_text_reply(
                conversation_id,
                final_reply.text,
                final_reply.delay_seconds,
                raw_payload=inbound_message.raw_payload,
                metadata={
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "trajectory": stage_decision.trajectory,
                    "delivery": "queued",
                },
            )
            self.conversation_repo.save_message(
                conversation_id,
                "ai",
                "text",
                final_reply.text,
                {
                    "selected_content": selected_content,
                    "turn_decision": asdict(turn_decision),
                    "style_spec": asdict(style_spec),
                    "trajectory": stage_decision.trajectory,
                    "delivery": send_result,
                },
                None,
            )
            self.conversation_repo.set_last_ai_reply_at(conversation_id)

        self.conversation_repo.update_conversation_state(
            conversation_id,
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding.current_mainline_should_continue,
        )
        if project_decision.project_id is not None:
            self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
        self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(
                business_account_id,
                user_id,
                ops_decision["ops_category"],
                ops_decision["reason"],
                "system",
            )
        try:
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "AI Turn Decision",
                {
                    "conversation_id": conversation_id,
                    "stage": stage_decision.stage,
                    "trajectory": stage_decision.trajectory,
                    "momentum": stage_decision.momentum,
                    "maturity_state": stage_decision.maturity_state,
                    "reply_goal": turn_decision.reply_goal,
                    "engagement_mode": turn_decision.engagement_mode,
                    "project_window": turn_decision.project_window,
                    "memory_priority": turn_decision.memory_priority,
                },
            )
        except Exception:
            logger.exception("failed to create AI turn decision receipt")

        if hasattr(self, "memory_writeback_engine") and self.memory_writeback_engine:
            self.memory_writeback_engine.write_turn(
                business_account_id,
                user_id,
                conversation_id,
                turn_decision,
                stage_decision,
                understanding,
                final_reply.text,
                latest_user_text,
            )
        if hasattr(self, "followup_scheduler") and self.followup_scheduler:
            self.followup_scheduler.schedule_after_turn(
                business_account_id,
                user_id,
                conversation_id,
                understanding,
                stage_decision,
                turn_decision,
                user_state,
            )
        if self.handover_learning_engine:
            self.handover_learning_engine.learn_after_turn(
                business_account_id,
                user_id,
                conversation_id,
                context,
                turn_decision,
                stage_decision,
                understanding,
            )
        if escalation_decision.get("should_queue_admin"):
            queue_type = "urgent_handover" if escalation_decision.get("notify_level") == "urgent_takeover" else "high_intent"
            priority_score = 95.0 if queue_type == "urgent_handover" else (80.0 if escalation_decision.get("notify_level") == "suggest_takeover" else 60.0)
            self.admin_queue_repo.upsert_queue_item(business_account_id, user_id, queue_type, priority_score, escalation_decision["reason"])
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "High intent detected",
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
                self.admin_notifier.notify_high_intent(
                    business_account_id,
                    user_id,
                    conversation_id,
                    escalation_decision.get("notify_level") or "watch",
                    escalation_decision.get("reason") or "",
                )


def build_app_components(settings: Settings) -> dict[str, Any]:
    db = Database(settings.database_url)
    db.connect()
    initialize_database(db)
    ensure_v1_upgrade_schema(db)
    ensure_v3_upgrade_schema(db)
    ensure_v4_upgrade_schema(db)
    ensure_v5_upgrade_schema(db)
    ensure_v51_upgrade_schema(db)

    tg_client = TelegramBotAPIClient(
        bot_token=settings.tg_bot_token,
        db=db,
        admin_chat_ids=settings.admin_chat_ids,
    )

    business_account_repo = BusinessAccountRepository(db)
    bootstrap_repo = BootstrapRepository(db)
    conversation_repo = ConversationRepository(db)
    decision_audit_repo = DecisionAuditRepository(db)
    override_lock_repo = OverrideLockRepository(db)
    strategy_learning_repo = StrategyLearningRepository(db)
    style_profile_repo = UserStyleProfileRepository(db)
    maturity_snapshot_repo = MaturitySnapshotRepository(db)
    manual_strategy_repo = ManualStrategyAbstractionRepository(db)
    user_repo = UserRepository(db, decision_audit_repo=decision_audit_repo)
    settings_repo = SettingsRepository(db)
    user_control_repo = UserControlRepository(db)
    material_repo = MaterialRepository(db)
    project_repo = ProjectRepository(db)
    script_repo = ScriptRepository(db)
    receipt_repo = ReceiptRepository(db)
    admin_queue_repo = AdminQueueRepository(db)
    handover_repo = HandoverRepository(db)
    processed_update_repo = ProcessedUpdateRepository(db)
    outbound_message_job_repo = OutboundMessageJobRepository(db)
    followup_repo = FollowupJobRepository(db)

    openai_adapter = OpenAIClientAdapter(settings.openai_api_key)
    llm_service = LLMService(openai_adapter, settings.llm_model_name)
    outbound_queue = OutboundMessageQueue(outbound_message_job_repo, tg_client)
    sender_service = SenderService(TelegramBusinessSenderAdapter(tg_client), outbound_queue=outbound_queue)
    admin_notifier = AdminNotifier(tg_client, admin_chat_ids=settings.admin_chat_ids)
    idempotency_guard = WebhookIdempotencyGuard(processed_update_repo)
    sender_worker = AsyncSenderWorker(settings.database_url, settings.tg_bot_token)
    receipt_center = ReceiptCenter(receipt_repo, decision_audit_repo)

    persona_core = PersonaCore()
    persona_profile_builder = PersonaProfileBuilder(material_repo)
    understanding_engine = UserUnderstandingEngine(llm_service)
    trajectory_engine = RelationshipTrajectoryEngine()
    maturity_engine = RelationshipMaturityEngine()
    stage_engine = ConversationStageEngine(trajectory_engine=trajectory_engine, maturity_engine=maturity_engine)
    mode_router = ChatModeRouter()
    reply_planner = ReplyPlanner()
    reply_style_engine = ReplyStyleEngine(llm_service)
    reply_self_check_engine = ReplySelfCheckEngine(polish_engine=MaturityPolishEngine())
    reply_delay_engine = ReplyDelayEngine()
    strategy_optimizer = AdaptiveStrategyOptimizer()
    rhythm_engine = RelationshipRhythmEngine()
    memory_priority_resolver = MemoryPriorityResolver()
    project_window_evaluator = ProjectWindowEvaluator()
    turn_decision_engine = TurnDecisionEngine(
        optimizer=strategy_optimizer,
        rhythm_engine=rhythm_engine,
        maturity_engine=maturity_engine,
        project_window_evaluator=project_window_evaluator,
        memory_resolver=memory_priority_resolver,
    )
    humanization_controller = HumanizationController(optimizer=strategy_optimizer)

    ai_switch_engine = AISwitchEngine(settings_repo, user_control_repo)
    project_classifier = ProjectClassifier(project_repo, override_lock_repo=override_lock_repo)
    project_segment_manager = ProjectSegmentManager(project_repo)
    tagging_engine = TaggingEngine()
    intent_engine = IntentEngine()
    human_escalation_engine = HumanEscalationEngine()
    ops_category_manager = OpsCategoryManager()

    content_selector = ContentSelector(material_repo, ScriptSelector(script_repo), MaterialSelector(material_repo), PersonaMaterialSelector())
    project_nurture_planner = ProjectNurturePlanner(window_evaluator=project_window_evaluator)
    memory_writeback_engine = MemoryWritebackEngine(
        conversation_repo,
        style_profile_repo=style_profile_repo,
        strategy_learning_repo=strategy_learning_repo,
        maturity_snapshot_repo=maturity_snapshot_repo,
    )
    followup_scheduler = FollowupScheduler(followup_repo, outbound_queue, receipt_repo)
    handover_learning_engine = HandoverLearningEngine(
        strategy_learning_repo,
        abstraction_repo=manual_strategy_repo,
        abstractor=ManualStrategyAbstractor(),
        resume_bridge_planner=ResumeBridgePlanner(),
    )

    handover_manager = HandoverManager(handover_repo, user_control_repo, conversation_repo, admin_queue_repo)
    handover_summary_builder = HandoverSummaryBuilder(handover_repo, conversation_repo, llm_service)
    resume_chat_manager = ResumeChatManager(handover_repo, conversation_repo, user_control_repo)

    customer_actions = CustomerActions(
        user_control_repo, user_repo, handover_manager, handover_summary_builder,
        resume_chat_manager, handover_repo, conversation_repo,
        receipt_center=receipt_center, override_lock_repo=override_lock_repo,
    )
    admin_api_service = AdminAPIService(
        user_repo, receipt_repo, handover_repo, conversation_repo, user_control_repo,
        admin_queue_repo, customer_actions, resume_chat_manager, project_repo,
        decision_audit_repo=decision_audit_repo, override_lock_repo=override_lock_repo,
    )
    dashboard_service = DashboardService(
        settings_repo, admin_queue_repo, receipt_repo, handover_repo, conversation_repo,
        decision_audit_repo=decision_audit_repo, override_lock_repo=override_lock_repo,
    )
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
        turn_decision_engine=turn_decision_engine,
        humanization_controller=humanization_controller,
        project_nurture_planner=project_nurture_planner,
        memory_writeback_engine=memory_writeback_engine,
        followup_scheduler=followup_scheduler,
        memory_selector=MemorySelector(resolver=memory_priority_resolver, guard=ContinuityGuard()),
        trajectory_engine=trajectory_engine,
        strategy_optimizer=strategy_optimizer,
        strategy_learning_repo=strategy_learning_repo,
        style_profile_repo=style_profile_repo,
        handover_learning_engine=handover_learning_engine,
        rhythm_engine=rhythm_engine,
        maturity_snapshot_repo=maturity_snapshot_repo,
        manual_strategy_repo=manual_strategy_repo,
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
        "idempotency_guard": idempotency_guard,
        "sender_worker": sender_worker,
        "maturity_snapshot_repo": maturity_snapshot_repo,
        "manual_strategy_repo": manual_strategy_repo,
    }


# ======== V5.1 continued optimization: stability, calibration, anchor continuity ========

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
    emotion_strength: float = 0.4
    boundary_strength: float = 0.0
    social_openness: float = 0.45
    project_relevance: float = 0.0
    human_takeover_hint: bool = False
    notes: list[str] = field(default_factory=list)
    hidden_intent: str | None = None
    conversation_energy: str = "steady"
    hesitation_score: float = 0.0
    style_signal: str = "balanced"
    relationship_readiness: str = "warmup"
    reassurance_need: float = 0.0
    continuity_need: float = 0.0
    pressure_risk: float = 0.0
    reluctance_type: str = "none"
    warmth_drift: str = "steady"


@dataclass
class StageDecision:
    stage: str
    changed: bool
    reason: str
    confidence: float = 0.65
    trajectory: str = "steady"
    momentum: str = "neutral"
    maturity_state: str = "emerging"
    maturity_reason: str = ""
    maturity_score: float = 0.5
    pressure_zone: str = "balanced"


@dataclass
class TurnDecision:
    reply_goal: str = "rapport"
    should_push_project: bool = False
    social_distance: str = "balanced"
    reply_length: str = "medium"
    ask_followup_question: bool = False
    exit_strategy: str = "gentle_close"
    self_disclosure_level: str = "light"
    reason: str = ""
    should_reply: bool = True
    marketing_intensity: str = "subtle"
    tone_bias: str = "natural"
    need_followup: bool = False
    followup_type: str | None = None
    followup_delay_seconds: int | None = None
    nurture_goal: str | None = None
    should_write_memory: bool = True
    trajectory_goal: str | None = None
    memory_focus: str | None = None
    user_style_profile: str = "balanced"
    project_strategy: str | None = None
    relationship_rhythm: str = "steady"
    engagement_mode: str = "maintain"
    recovery_window_hours: int = 24
    trust_signal: str = "neutral"
    maturity_target: str = "stable_presence"
    memory_priority: str = "relationship"
    continuity_mode: str = "implicit"
    project_window: str = "hold"
    maturity_score: float = 0.5
    calibration_mode: str = "balanced"
    anchor_memory_hint: str | None = None
    pressure_guard: str = "normal"
    window_confidence: float = 0.5


@dataclass
class StyleSpec:
    tone: str = "natural"
    formality: str = "balanced"
    warmth: str = "medium"
    length: str = "medium"
    question_rate: str = "low"
    emoji_usage: str = "minimal"
    self_disclosure_ratio: str = "none"
    completion_level: str = "balanced"
    marketing_visibility: str = "subtle"
    naturalness_bias: str = "high"
    cadence_bias: str = "steady"
    initiative_level: str = "low"
    wording_texture: str = "smooth"
    pause_texture: str = "natural"
    anchor_style: str = "light"
    maturity_polish: str = "balanced"
    directness: str = "balanced"
    sentence_density: str = "balanced"
    softener_level: str = "balanced"


class RhythmConflictResolver:
    def resolve(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        rhythm_data: dict[str, Any],
        maturity_data: dict[str, Any],
        stored_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stored_profile = stored_profile or {}
        rhythm = str(rhythm_data.get("relationship_rhythm") or rhythm_data.get("rhythm") or "steady")
        engagement_mode = str(rhythm_data.get("engagement_mode") or "maintain")
        trust_signal = str(rhythm_data.get("trust_signal") or "neutral")
        maturity_state = str(maturity_data.get("maturity_state") or maturity_data.get("label") or stage_decision.maturity_state or "emerging")
        pressure_zone = "balanced"
        calibration_mode = "balanced"
        pressure_guard = "normal"

        if understanding.boundary_signal or understanding.boundary_strength > 0.7:
            pressure_zone = "guarded"
            calibration_mode = "space_first"
            pressure_guard = "strict"
            engagement_mode = "protect_space"
            rhythm = "slow"
            trust_signal = "fragile"
        elif understanding.pressure_risk > 0.7:
            pressure_zone = "high_risk"
            calibration_mode = "reduce_pressure"
            pressure_guard = "strict"
            if engagement_mode == "clarify_and_progress":
                engagement_mode = "reassure_and_hold"
            rhythm = "patient" if rhythm in ("responsive", "warm") else rhythm
        elif maturity_state in ("fragile_space", "careful_growth") and engagement_mode == "clarify_and_progress":
            pressure_zone = "careful"
            calibration_mode = "careful_open"
            pressure_guard = "watch"
            engagement_mode = "reassure_and_hold"
        elif maturity_state in ("stable_building", "mature_progress") and stage_decision.trajectory in ("conversion_window", "project_deepen"):
            pressure_zone = "open"
            calibration_mode = "clear_progress"
            pressure_guard = "normal"
        elif understanding.warmth_drift == "cooling":
            pressure_zone = "cooling"
            calibration_mode = "reconnect_lightly"
            pressure_guard = "watch"
            if engagement_mode == "bond_and_expand":
                engagement_mode = "maintain"

        if (stored_profile.get("style_profile") or "") == "space_preferring" and pressure_guard != "strict":
            pressure_guard = "watch"
            calibration_mode = "space_first"
            rhythm = "slow"
            if engagement_mode in ("clarify_and_progress", "bond_and_expand"):
                engagement_mode = "maintain"

        return {
            "relationship_rhythm": rhythm,
            "engagement_mode": engagement_mode,
            "trust_signal": trust_signal,
            "pressure_zone": pressure_zone,
            "calibration_mode": calibration_mode,
            "pressure_guard": pressure_guard,
        }


_Prev_MemoryPriorityResolver_1 = MemoryPriorityResolver
class MemoryPriorityResolver(_Prev_MemoryPriorityResolver_1):
    def resolve(
        self,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        turn_preview: dict[str, Any] | None = None,
        stored_profile: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        base = super().resolve(understanding, stage_decision, turn_preview=turn_preview, stored_profile=stored_profile)
        priority = base.get("memory_priority", "relationship")
        continuity_mode = base.get("continuity_mode", "implicit")
        if understanding.reluctance_type in ("soft_resistance", "needs_space"):
            priority = "boundary"
            continuity_mode = "light_touch"
        elif understanding.warmth_drift == "cooling" and priority == "project_clarity":
            priority = "relationship"
            continuity_mode = "soft_reference"
        elif understanding.relationship_readiness in ("trust_window", "ready_for_clarity") and understanding.explicit_product_query:
            priority = "project_clarity"
            continuity_mode = "fact_link"
        return {"memory_priority": priority, "continuity_mode": continuity_mode}


_Prev_ContinuityGuard_1 = ContinuityGuard
class ContinuityGuard(_Prev_ContinuityGuard_1):
    def trim_recent(self, selected_recent: list[dict], continuity_mode: str, priority: str) -> list[dict]:
        msgs = super().trim_recent(selected_recent, continuity_mode, priority)
        if continuity_mode == "implicit":
            return msgs[-4:]
        if priority == "boundary":
            return msgs[-3:]
        return msgs

    def should_surface_summary(self, understanding: UnderstandingResult, priority: str) -> bool:
        if understanding.reluctance_type == "needs_space":
            return False
        return super().should_surface_summary(understanding, priority)


class AnchorMemoryEngine:
    def pick_anchor(
        self,
        selected_recent: list[dict],
        medium_summary: str | None,
        continuity_mode: str,
        priority: str,
    ) -> str | None:
        if priority == "boundary" or continuity_mode == "light_touch":
            return None
        for msg in reversed(selected_recent or []):
            txt = str(msg.get("text") or "").strip()
            if len(txt) >= 16:
                return txt[:140]
        if medium_summary and continuity_mode in ("gentle_reference", "soft_reference", "fact_link"):
            return str(medium_summary)[:140]
        return None


class StyleDampingEngine:
    def stabilize(self, style_spec: StyleSpec, turn_decision: TurnDecision, stored_profile: dict[str, Any] | None = None) -> StyleSpec:
        stored_profile = stored_profile or {}
        if turn_decision.pressure_guard == "strict":
            style_spec.question_rate = "minimal"
            style_spec.initiative_level = "low"
            style_spec.length = "short" if style_spec.length == "medium" else style_spec.length
            style_spec.sentence_density = "light"
            style_spec.softener_level = "high"
        elif turn_decision.calibration_mode == "clear_progress":
            style_spec.directness = "balanced" if style_spec.directness == "soft" else style_spec.directness
            style_spec.sentence_density = "lean"
            style_spec.softener_level = "low"
        else:
            style_spec.sentence_density = "balanced"
            style_spec.softener_level = "balanced"
        if (stored_profile.get("style_profile") or "") == "logic_first":
            style_spec.anchor_style = "fact"
        elif (stored_profile.get("style_profile") or "") == "emotion_first":
            style_spec.anchor_style = "soft"
        return style_spec


_Prev_ProjectWindowEvaluator_1 = ProjectWindowEvaluator
class ProjectWindowEvaluator(_Prev_ProjectWindowEvaluator_1):
    def evaluate(self, understanding: UnderstandingResult, stage_decision: StageDecision, turn_decision: TurnDecision) -> str:
        base = super().evaluate(understanding, stage_decision, turn_decision)
        if turn_decision.pressure_guard == "strict":
            return "hold"
        if understanding.warmth_drift == "cooling" and base in ("open", "careful_open"):
            return "soft_background"
        if turn_decision.calibration_mode == "clear_progress" and base == "careful_open":
            return "open"
        return base


_Prev_ManualStrategyAbstractor_1 = ManualStrategyAbstractor
class ManualStrategyAbstractor(_Prev_ManualStrategyAbstractor_1):
    def abstract(self, context: ConversationContext, turn_decision: TurnDecision, stage_decision: StageDecision, understanding: UnderstandingResult) -> dict[str, Any]:
        base = super().abstract(context, turn_decision, stage_decision, understanding)
        abstraction = dict(base or {})
        if turn_decision.pressure_guard == "strict":
            abstraction["abstraction_type"] = "space_protection"
        elif turn_decision.calibration_mode == "clear_progress":
            abstraction["abstraction_type"] = "clear_progress"
        abstraction["abstraction_text"] = (abstraction.get("abstraction_text") or "") + f"; pressure_guard={turn_decision.pressure_guard}; calibration={turn_decision.calibration_mode}"
        return abstraction


_Prev_ResumeBridgePlanner_1 = ResumeBridgePlanner
class ResumeBridgePlanner(_Prev_ResumeBridgePlanner_1):
    def build(self, turn_decision: TurnDecision, understanding: UnderstandingResult) -> str:
        base = super().build(turn_decision, understanding)
        if turn_decision.pressure_guard == "strict":
            return "return softly with low-pressure acknowledgment"
        if understanding.warmth_drift == "cooling":
            return "re-enter with light continuity and no project push"
        return base


_Prev_MaturityPolishEngine_1 = MaturityPolishEngine
class MaturityPolishEngine(_Prev_MaturityPolishEngine_1):
    def polish(self, text: str, understanding: UnderstandingResult, style_spec: StyleSpec, turn_decision: TurnDecision) -> str:
        t = super().polish(text, understanding, style_spec, turn_decision)
        if not t:
            return t
        if turn_decision.pressure_guard == "strict":
            t = re.sub(r'\b(?:whenever you want|if you want to|when you are ready)\b', '', t, flags=re.IGNORECASE)
        if style_spec.sentence_density == "light" and len(t) > 220:
            t = t[:220].rsplit(' ', 1)[0].rstrip(' ,') + '.'
        if style_spec.anchor_style == "fact":
            t = re.sub(r'\bkind of\b', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s{2,}', ' ', t).strip()
        return t


_Prev_UserUnderstandingEngine_5 = UserUnderstandingEngine
class UserUnderstandingEngine(_Prev_UserUnderstandingEngine_5):
    def analyze(self, latest_user_text: str, recent_context: list[dict], persona_summary: str, user_state_summary: str) -> UnderstandingResult:
        result = super().analyze(latest_user_text, recent_context, persona_summary, user_state_summary)
        data = asdict(result)
        text = (latest_user_text or "").lower()
        reluctance_type = "none"
        warmth_drift = "steady"
        if any(k in text for k in ["maybe later", "not now", "a bit much", "too much", "i need space"]):
            reluctance_type = "needs_space"
        elif any(k in text for k in ["not sure", "maybe", "i guess", "i'm thinking", "let me think"]):
            reluctance_type = "soft_resistance"
        elif any(k in text for k in ["how exactly", "what do you mean", "can you clarify", "what would that look like"]):
            reluctance_type = "clarity_seek"
        if any(k in text for k in ["busy", "later", "talk tomorrow", "another time"]):
            warmth_drift = "cooling"
        elif any(k in text for k in ["sounds good", "that helps", "i get you", "that's fair"]):
            warmth_drift = "warming"
        data["reluctance_type"] = reluctance_type
        data["warmth_drift"] = warmth_drift
        return UnderstandingResult(**data)


_Prev_AdaptiveStrategyOptimizer_3 = AdaptiveStrategyOptimizer
class AdaptiveStrategyOptimizer(_Prev_AdaptiveStrategyOptimizer_3):
    def infer_user_style(
        self,
        understanding: UnderstandingResult,
        user_state: UserStateSnapshot,
        stored_profile: dict[str, Any] | None = None,
        recent_learning: list[dict] | None = None,
    ) -> str:
        style = super().infer_user_style(understanding, user_state, stored_profile=stored_profile, recent_learning=recent_learning)
        if understanding.reluctance_type == "needs_space":
            return "space_preferring"
        if understanding.reluctance_type == "clarity_seek" and style == "balanced":
            return "logic_first"
        return style


_Prev_MemorySelector_3 = MemorySelector
class MemorySelector(_Prev_MemorySelector_3):
    def __init__(self, resolver: MemoryPriorityResolver | None = None, guard: ContinuityGuard | None = None, anchor_engine: AnchorMemoryEngine | None = None) -> None:
        super().__init__(resolver=resolver or MemoryPriorityResolver(), guard=guard or ContinuityGuard())
        self.anchor_engine = anchor_engine or AnchorMemoryEngine()

    def select(
        self,
        memory_bundle: MemoryBundle,
        understanding: UnderstandingResult,
        trajectory: str,
        stored_profile: dict[str, Any] | None = None,
        recent_learning: list[dict] | None = None,
        stage_decision: StageDecision | None = None,
    ) -> dict[str, Any]:
        base = super().select(memory_bundle, understanding, trajectory, stored_profile=stored_profile, recent_learning=recent_learning, stage_decision=stage_decision)
        anchor = self.anchor_engine.pick_anchor(
            base.get("selected_recent") or [],
            base.get("selected_summary"),
            base.get("continuity_mode") or "implicit",
            base.get("memory_focus") or "relationship",
        )
        base["anchor_memory_hint"] = anchor
        return base


_Prev_TurnDecisionEngine_7 = TurnDecisionEngine
class TurnDecisionEngine(_Prev_TurnDecisionEngine_7):
    def __init__(
        self,
        optimizer: AdaptiveStrategyOptimizer | None = None,
        rhythm_engine: RelationshipRhythmEngine | None = None,
        maturity_engine: RelationshipMaturityEngine | None = None,
        project_window_evaluator: ProjectWindowEvaluator | None = None,
        memory_resolver: MemoryPriorityResolver | None = None,
        conflict_resolver: RhythmConflictResolver | None = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer or AdaptiveStrategyOptimizer(),
            rhythm_engine=rhythm_engine or RelationshipRhythmEngine(),
            maturity_engine=maturity_engine or RelationshipMaturityEngine(),
            project_window_evaluator=project_window_evaluator or ProjectWindowEvaluator(),
            memory_resolver=memory_resolver or MemoryPriorityResolver(),
        )
        self.conflict_resolver = conflict_resolver or RhythmConflictResolver()

    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        decision = super().decide(understanding, stage_decision, mode_decision, user_state)
        stored_profile = getattr(user_state, "_style_profile_data", None) or {}
        rhythm_data = {
            "relationship_rhythm": decision.relationship_rhythm,
            "engagement_mode": decision.engagement_mode,
            "trust_signal": decision.trust_signal,
        }
        maturity_data = {
            "maturity_state": decision.maturity_target,
            "maturity_score": decision.maturity_score,
        }
        resolved = self.conflict_resolver.resolve(understanding, stage_decision, rhythm_data, maturity_data, stored_profile=stored_profile)
        data = {**asdict(decision)}
        data["relationship_rhythm"] = resolved["relationship_rhythm"]
        data["engagement_mode"] = resolved["engagement_mode"]
        data["trust_signal"] = resolved["trust_signal"]
        data["calibration_mode"] = resolved["calibration_mode"]
        data["pressure_guard"] = resolved["pressure_guard"]
        data["window_confidence"] = 0.78 if data["project_window"] in ("open", "careful_open") else 0.52
        if understanding.reluctance_type == "needs_space":
            data["reply_length"] = "short"
            data["ask_followup_question"] = False
            data["should_push_project"] = False
            data["project_strategy"] = "hold_with_space"
        elif understanding.reluctance_type == "clarity_seek" and data["project_window"] in ("careful_open", "open"):
            data["project_strategy"] = "clarify_then_pause"
        data["memory_focus"] = data.get("memory_priority")
        return TurnDecision(**data)


_Prev_HumanizationController_6 = HumanizationController
class HumanizationController(_Prev_HumanizationController_6):
    def __init__(self, optimizer: AdaptiveStrategyOptimizer | None = None, damping_engine: StyleDampingEngine | None = None) -> None:
        super().__init__(optimizer=optimizer or AdaptiveStrategyOptimizer())
        self.damping_engine = damping_engine or StyleDampingEngine()

    def build_style_spec(
        self,
        decision: TurnDecision,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        persona_profile: dict[str, Any] | None = None,
    ) -> StyleSpec:
        spec = super().build_style_spec(decision, understanding, stage_decision, user_state, persona_profile)
        if decision.calibration_mode == "space_first":
            spec.pause_texture = "measured"
            spec.anchor_style = "implicit"
        elif decision.calibration_mode == "clear_progress":
            spec.pause_texture = "clean"
            spec.anchor_style = "fact"
        elif understanding.warmth_drift == "warming":
            spec.pause_texture = "soft"
            spec.anchor_style = "soft"
        stored_profile = getattr(user_state, "_style_profile_data", None) or {}
        spec = self.damping_engine.stabilize(spec, decision, stored_profile=stored_profile)
        return spec


_Prev_ProjectNurturePlanner_4 = ProjectNurturePlanner
class ProjectNurturePlanner(_Prev_ProjectNurturePlanner_4):
    def plan(self, turn_decision: TurnDecision, understanding: UnderstandingResult, stage_decision: StageDecision, user_state: UserStateSnapshot) -> dict[str, Any]:
        plan = super().plan(turn_decision, understanding, stage_decision, user_state)
        if turn_decision.project_window == "hold" or turn_decision.pressure_guard == "strict":
            plan["angle"] = "presence_over_project"
            plan["visibility"] = "none"
        elif turn_decision.project_strategy == "clarify_then_pause":
            plan["angle"] = "clear_fact_then_pause"
            plan["visibility"] = "light"
        return plan


_Prev_MemoryWritebackEngine_4 = MemoryWritebackEngine
class MemoryWritebackEngine(_Prev_MemoryWritebackEngine_4):
    def write_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        turn_decision: TurnDecision,
        stage_decision: StageDecision,
        understanding: UnderstandingResult,
        final_reply_text: str,
        latest_user_text: str,
    ) -> None:
        super().write_turn(business_account_id, user_id, conversation_id, turn_decision, stage_decision, understanding, final_reply_text, latest_user_text)
        if self.strategy_learning_repo:
            self.strategy_learning_repo.insert_learning(
                business_account_id,
                user_id,
                conversation_id,
                stage_decision.trajectory,
                turn_decision.user_style_profile,
                turn_decision.reply_goal,
                f"calibration={turn_decision.calibration_mode}; pressure_guard={turn_decision.pressure_guard}; memory_focus={turn_decision.memory_focus}",
                "stabilized_turn",
            )


_Prev_HandoverLearningEngine_3 = HandoverLearningEngine
class HandoverLearningEngine(_Prev_HandoverLearningEngine_3):
    def learn_after_turn(
        self,
        business_account_id: int,
        user_id: int,
        conversation_id: int,
        context: ConversationContext,
        turn_decision: TurnDecision,
        stage_decision: StageDecision,
        understanding: UnderstandingResult,
    ) -> None:
        super().learn_after_turn(business_account_id, user_id, conversation_id, context, turn_decision, stage_decision, understanding)
        if self.abstraction_repo:
            self.abstraction_repo.insert_abstraction(
                business_account_id,
                user_id,
                conversation_id,
                "calibration_trace",
                f"calibration={turn_decision.calibration_mode}; pressure_guard={turn_decision.pressure_guard}; anchor={turn_decision.anchor_memory_hint or 'none'}",
                understanding.reluctance_type or "none",
            )


_Prev_ReplySelfCheckEngine_5 = ReplySelfCheckEngine
class ReplySelfCheckEngine(_Prev_ReplySelfCheckEngine_5):
    def __init__(self, polish_engine: MaturityPolishEngine | None = None) -> None:
        super().__init__(polish_engine=polish_engine or MaturityPolishEngine())

    def check_and_fix_with_context(
        self,
        draft_reply: str,
        mode: str,
        understanding: UnderstandingResult,
        style_spec: StyleSpec,
        turn_decision: TurnDecision,
    ) -> str:
        text = super().check_and_fix_with_context(draft_reply, mode, understanding, style_spec, turn_decision)
        if turn_decision.pressure_guard == "strict" and turn_decision.anchor_memory_hint and len(text) > 260:
            text = text[:260].rsplit(' ', 1)[0].rstrip(' ,') + '.'
        if style_spec.anchor_style == "implicit":
            text = re.sub(r'\bremember\b', 'it', text, flags=re.IGNORECASE)
        return " ".join(text.split()).strip()


_old_build_app_components_v51 = build_app_components

def build_app_components(settings: Settings) -> dict[str, Any]:
    components = _old_build_app_components_v51(settings)
    strategy_optimizer = AdaptiveStrategyOptimizer()
    rhythm_engine = RelationshipRhythmEngine()
    maturity_engine = RelationshipMaturityEngine()
    memory_priority_resolver = MemoryPriorityResolver()
    project_window_evaluator = ProjectWindowEvaluator()
    conflict_resolver = RhythmConflictResolver()
    damping_engine = StyleDampingEngine()
    turn_decision_engine = TurnDecisionEngine(
        optimizer=strategy_optimizer,
        rhythm_engine=rhythm_engine,
        maturity_engine=maturity_engine,
        project_window_evaluator=project_window_evaluator,
        memory_resolver=memory_priority_resolver,
        conflict_resolver=conflict_resolver,
    )
    humanization_controller = HumanizationController(optimizer=strategy_optimizer, damping_engine=damping_engine)
    memory_selector = MemorySelector(resolver=memory_priority_resolver, guard=ContinuityGuard(), anchor_engine=AnchorMemoryEngine())
    project_nurture_planner = ProjectNurturePlanner(window_evaluator=project_window_evaluator)
    orchestrator = components.get("orchestrator")
    if orchestrator:
        orchestrator.turn_decision_engine = turn_decision_engine
        orchestrator.humanization_controller = humanization_controller
        orchestrator.memory_selector = memory_selector
        orchestrator.project_nurture_planner = project_nurture_planner
        orchestrator.strategy_optimizer = strategy_optimizer
        orchestrator.rhythm_engine = rhythm_engine
    components["turn_decision_engine"] = turn_decision_engine
    components["humanization_controller"] = humanization_controller
    components["memory_selector"] = memory_selector
    components["project_nurture_planner"] = project_nurture_planner
    return components

if __name__ == '__main__':
    main()
