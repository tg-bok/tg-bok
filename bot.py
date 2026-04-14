from __future__ import annotations

import json
import logging
import os
import sys
import time
import re
import hmac
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
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
class MemoryBundle:
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    recent_summary: str = ""
    long_term_memory: dict[str, Any] = field(default_factory=dict)
    handover_summary: dict[str, Any] = field(default_factory=dict)



# ===== TurnDecision (10017-10053) =====
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



# ===== StyleSpec (10054-10076) =====
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



# ===== UnderstandingResult (9971-10002) =====
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



# ===== StageDecision (10003-10016) =====
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



# ===== UserStateSnapshot (4481-4502) =====
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

    def get_persona_materials(self, business_account_id: int, material_type: str | None = None) -> list[dict]:
        with self.db.cursor() as cur:
            if material_type:
                cur.execute(
                    "SELECT * FROM persona_materials WHERE business_account_id=%s AND is_active=TRUE AND material_type=%s ORDER BY priority ASC, id ASC",
                    (business_account_id, material_type),
                )
            else:
                cur.execute(
                    "SELECT * FROM persona_materials WHERE business_account_id=%s AND is_active=TRUE ORDER BY priority ASC, id ASC",
                    (business_account_id,),
                )
            return cur.fetchall()

    def create_persona_material(self, business_account_id: int, material_type: str, content_text: str | None = None, media_url: str | None = None, scene_tags_json: list | None = None, priority: int = 100) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO persona_materials (business_account_id, material_type, content_text, media_url, scene_tags_json, priority, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, TRUE, NOW(), NOW())
                    RETURNING *
                    """,
                    (business_account_id, material_type, content_text, media_url, Json(scene_tags_json or []), priority),
                )
                return cur.fetchone() or {}

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

    def create_project_material(self, project_id: int, material_type: str, content_text: str | None = None, media_url: str | None = None, scene_tags_json: list | None = None, priority: int = 100) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO project_materials (project_id, material_type, content_text, media_url, scene_tags_json, priority, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, TRUE, NOW(), NOW())
                    RETURNING *
                    """,
                    (project_id, material_type, content_text, media_url, Json(scene_tags_json or []), priority),
                )
                return cur.fetchone() or {}


class ProjectRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def list_active_projects(self, business_account_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT *, CASE WHEN is_active THEN 'active' ELSE 'inactive' END AS status FROM projects WHERE business_account_id=%s ORDER BY id ASC",
                (business_account_id,),
            )
            return cur.fetchall()

    def create_project(self, business_account_id: int, name: str, description: str | None = None) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projects (business_account_id, name, description, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, TRUE, NOW(), NOW())
                    RETURNING *
                    """,
                    (business_account_id, name, description),
                )
                return cur.fetchone() or {}

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

    def create_project_script(self, project_id: int, category: str | None, content_text: str, priority: int = 100) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO project_scripts (project_id, category, content_text, priority, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, TRUE, NOW(), NOW())
                    RETURNING *
                    """,
                    (project_id, category, content_text, priority),
                )
                return cur.fetchone() or {}


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

    def list_pending_receipts(self, business_account_id: int, limit: int = 20) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM receipts WHERE business_account_id=%s AND status='pending' ORDER BY created_at DESC LIMIT %s",
                (business_account_id, limit),
            )
            return cur.fetchall()


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
        self.material_repo = material_repo
        self.script_repo = script_repo

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
        self.material_repo = material_repo
        self.script_repo = script_repo

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
        self.material_repo = material_repo
        self.script_repo = script_repo
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




# ==== Rebuild Stage2 advanced core modules ====
# ===== base ProjectNurturePlanner (5319-5352) =====
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



# ===== latest ProjectNurturePlanner (10412-10423) =====
class ProjectNurturePlanner(ProjectNurturePlanner):
    def plan(self, turn_decision: TurnDecision, understanding: UnderstandingResult, stage_decision: StageDecision, user_state: UserStateSnapshot) -> dict[str, Any]:
        plan = super().plan(turn_decision, understanding, stage_decision, user_state)
        if turn_decision.project_window == "hold" or turn_decision.pressure_guard == "strict":
            plan["angle"] = "presence_over_project"
            plan["visibility"] = "none"
        elif turn_decision.project_strategy == "clarify_then_pause":
            plan["angle"] = "clear_fact_then_pause"
            plan["visibility"] = "light"
        return plan



# ===== class RelationshipTrajectoryEngine =====
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



# ===== class RelationshipRhythmEngine =====
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



# ===== class RelationshipMaturityEngine =====
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



# ===== class RhythmConflictResolver =====
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

# ===== base TurnDecisionEngine (3907-3964) =====
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



# ===== latest TurnDecisionEngine (10334-10383) =====
class TurnDecisionEngine(TurnDecisionEngine):
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



# ===== base MemorySelector (7019-7047) =====
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



# ===== latest MemorySelector (10309-10333) =====
class MemorySelector(MemorySelector):
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



# ===== base MemoryPriorityResolver (8982-9007) =====
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



# ===== latest MemoryPriorityResolver (10142-10164) =====
class MemoryPriorityResolver(MemoryPriorityResolver):
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



# ===== base ContinuityGuard (9008-9026) =====
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



# ===== latest ContinuityGuard (10165-10179) =====
class ContinuityGuard(ContinuityGuard):
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



# ===== base AdaptiveStrategyOptimizer (7048-7069) =====
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



# ===== latest AdaptiveStrategyOptimizer (10293-10308) =====
class AdaptiveStrategyOptimizer(AdaptiveStrategyOptimizer):
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



# ===== base HumanizationController (3350-3409) =====
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



# ===== latest HumanizationController (10384-10411) =====
class HumanizationController(HumanizationController):
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



# ===== base ProjectWindowEvaluator (9041-9053) =====
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



# ===== latest ProjectWindowEvaluator (10222-10233) =====
class ProjectWindowEvaluator(ProjectWindowEvaluator):
    def evaluate(self, understanding: UnderstandingResult, stage_decision: StageDecision, turn_decision: TurnDecision) -> str:
        base = super().evaluate(understanding, stage_decision, turn_decision)
        if turn_decision.pressure_guard == "strict":
            return "hold"
        if understanding.warmth_drift == "cooling" and base in ("open", "careful_open"):
            return "soft_background"
        if turn_decision.calibration_mode == "clear_progress" and base == "careful_open":
            return "open"
        return base



# ===== base MaturityPolishEngine (9084-9117) =====
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



# ===== latest MaturityPolishEngine (10256-10270) =====
class MaturityPolishEngine(MaturityPolishEngine):
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



# ===== base MemoryWritebackEngine (5353-5378) =====
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



# ===== latest MemoryWritebackEngine (10424-10449) =====
class MemoryWritebackEngine(MemoryWritebackEngine):
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



# ===== base HandoverLearningEngine (7070-7104) =====
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



# ===== latest HandoverLearningEngine (10450-10472) =====
class HandoverLearningEngine(HandoverLearningEngine):
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



# ===== base ManualStrategyAbstractor (9054-9072) =====
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



# ===== latest ManualStrategyAbstractor (10234-10245) =====
class ManualStrategyAbstractor(ManualStrategyAbstractor):
    def abstract(self, context: ConversationContext, turn_decision: TurnDecision, stage_decision: StageDecision, understanding: UnderstandingResult) -> dict[str, Any]:
        base = super().abstract(context, turn_decision, stage_decision, understanding)
        abstraction = dict(base or {})
        if turn_decision.pressure_guard == "strict":
            abstraction["abstraction_type"] = "space_protection"
        elif turn_decision.calibration_mode == "clear_progress":
            abstraction["abstraction_type"] = "clear_progress"
        abstraction["abstraction_text"] = (abstraction.get("abstraction_text") or "") + f"; pressure_guard={turn_decision.pressure_guard}; calibration={turn_decision.calibration_mode}"
        return abstraction



# ===== base ResumeBridgePlanner (9073-9083) =====
class ResumeBridgePlanner:
    def build(self, turn_decision: TurnDecision, understanding: UnderstandingResult) -> str:
        if turn_decision.engagement_mode == "protect_space":
            return "resume_with_acknowledgment_only"
        if understanding.reassurance_need > 0.58:
            return "resume_with_gentle_reassurance"
        if turn_decision.project_window in ("open", "careful_open"):
            return "resume_with_light_clarity"
        return "resume_with_soft_presence"



# ===== latest ResumeBridgePlanner (10246-10255) =====
class ResumeBridgePlanner(ResumeBridgePlanner):
    def build(self, turn_decision: TurnDecision, understanding: UnderstandingResult) -> str:
        base = super().build(turn_decision, understanding)
        if turn_decision.pressure_guard == "strict":
            return "return softly with low-pressure acknowledgment"
        if understanding.warmth_drift == "cooling":
            return "re-enter with light continuity and no project push"
        return base





class AdminAPIService:
    def __init__(self, user_repo: UserRepository, receipt_repo: ReceiptRepository, handover_repo: HandoverRepository, conversation_repo: ConversationRepository, user_control_repo: UserControlRepository, admin_queue_repo: AdminQueueRepository, customer_actions: CustomerActions, resume_chat_manager: ResumeChatManager, project_repo: ProjectRepository | None = None, material_repo: MaterialRepository | None = None, script_repo: ScriptRepository | None = None) -> None:
        self.user_repo = user_repo
        self.receipt_repo = receipt_repo
        self.handover_repo = handover_repo
        self.conversation_repo = conversation_repo
        self.user_control_repo = user_control_repo
        self.admin_queue_repo = admin_queue_repo
        self.customer_actions = customer_actions
        self.resume_chat_manager = resume_chat_manager
        self.project_repo = project_repo
        self.material_repo = material_repo
        self.script_repo = script_repo

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

    def create_project(self, business_account_id: int, name: str, description: str | None = None) -> dict:
        if not self.project_repo:
            return {"ok": False, "reason": "project repository unavailable"}
        row = self.project_repo.create_project(business_account_id, name, description)
        return {"ok": True, "project": row}

    def list_project_materials(self, project_id: int, material_type: str | None = None) -> list[dict]:
        if not hasattr(self, "material_repo") or self.material_repo is None:
            return []
        return self.material_repo.get_project_materials(project_id, material_type)

    def create_project_material(self, project_id: int, material_type: str, content_text: str | None = None, media_url: str | None = None, scene_tags_json: list | None = None, priority: int = 100) -> dict:
        if not hasattr(self, "material_repo") or self.material_repo is None:
            return {"ok": False, "reason": "material repository unavailable"}
        row = self.material_repo.create_project_material(project_id, material_type, content_text, media_url, scene_tags_json, priority)
        return {"ok": True, "material": row}

    def list_persona_materials(self, business_account_id: int, material_type: str | None = None) -> list[dict]:
        if not hasattr(self, "material_repo") or self.material_repo is None:
            return []
        return self.material_repo.get_persona_materials(business_account_id, material_type)

    def create_persona_material(self, business_account_id: int, material_type: str, content_text: str | None = None, media_url: str | None = None, scene_tags_json: list | None = None, priority: int = 100) -> dict:
        if not hasattr(self, "material_repo") or self.material_repo is None:
            return {"ok": False, "reason": "material repository unavailable"}
        row = self.material_repo.create_persona_material(business_account_id, material_type, content_text, media_url, scene_tags_json, priority)
        return {"ok": True, "material": row}

    def list_project_scripts(self, project_id: int, categories: list[str] | None = None) -> list[dict]:
        if not hasattr(self, "script_repo") or self.script_repo is None:
            return []
        return self.script_repo.get_project_scripts(project_id, categories)

    def create_project_script(self, project_id: int, category: str | None, content_text: str, priority: int = 100) -> dict:
        if not hasattr(self, "script_repo") or self.script_repo is None:
            return {"ok": False, "reason": "script repository unavailable"}
        row = self.script_repo.create_project_script(project_id, category, content_text, priority)
        return {"ok": True, "script": row}

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
    def _back_row(target: str = "adm:main") -> list[dict]:
        return [{"text": "⬅️ 返回", "callback_data": target}]

    @staticmethod
    def main_menu() -> dict:
        return {
            "text": "📊 管理后台\n请选择操作：",
            "reply_markup": [
                [{"text": "📈 仪表盘", "callback_data": "adm:dashboard"}, {"text": "👥 用户管理", "callback_data": "adm:users"}],
                [{"text": "🏷️ 用户分类", "callback_data": "adm:classification"}, {"text": "📦 项目管理", "callback_data": "adm:projects"}],
                [{"text": "🗂️ 素材管理", "callback_data": "adm:materials"}, {"text": "🧬 人设素材", "callback_data": "adm:persona_materials"}],
                [{"text": "📝 脚本管理", "callback_data": "adm:scripts"}, {"text": "🤝 接管管理", "callback_data": "adm:handover"}],
                [{"text": "📬 队列中心", "callback_data": "adm:queues"}, {"text": "🧾 回执审核", "callback_data": "adm:receipts"}],
                [{"text": "⚙️ 系统设置", "callback_data": "adm:settings"}],
            ],
        }

    @staticmethod
    def users_menu() -> dict:
        return {
            "text": "👥 用户管理\n请选择功能：",
            "reply_markup": [
                [{"text": "📋 用户列表", "callback_data": "adm:users:list"}, {"text": "🔥 高意向用户", "callback_data": "adm:queue:high_intent"}],
                [{"text": "🧯 接管中用户", "callback_data": "adm:handover:active"}, {"text": "⏳ 待恢复用户", "callback_data": "adm:queue:pending_resume"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }

    @staticmethod
    def classification_menu() -> dict:
        return {
            "text": "🏷️ 用户分类\n请选择功能：",
            "reply_markup": [
                [{"text": "📌 项目分类", "callback_data": "adm:classification:project"}, {"text": "🏷️ 标签分类", "callback_data": "adm:classification:tags"}],
                [{"text": "🧭 运营分类", "callback_data": "adm:classification:ops"}, {"text": "🔒 锁定管理", "callback_data": "adm:classification:locks"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }

    @staticmethod
    def projects_menu() -> dict:
        return {
            "text": "📦 项目管理\n请选择功能：",
            "reply_markup": [
                [{"text": "📚 项目列表", "callback_data": "adm:projects:list"}, {"text": "➕ 添加项目", "callback_data": "adm:projects:create"}],
                [{"text": "👤 用户项目绑定", "callback_data": "adm:projects:user_binding"}, {"text": "🔒 项目锁定", "callback_data": "adm:projects:locks"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }

    @staticmethod
    def materials_menu() -> dict:
        return {
            "text": "🗂️ 素材管理\n请选择素材类型：",
            "reply_markup": [
                [{"text": "📦 项目素材", "callback_data": "adm:materials:project"}, {"text": "🧬 人设素材", "callback_data": "adm:materials:persona"}],
                [{"text": "🌤️ 日常素材", "callback_data": "adm:materials:daily"}, {"text": "➕ 添加素材", "callback_data": "adm:materials:create"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }

    @staticmethod
    def persona_materials_menu() -> dict:
        return {
            "text": "🧬 人设素材\n请选择分类：",
            "reply_markup": [
                [{"text": "👤 基础身份", "callback_data": "adm:persona_materials:identity"}, {"text": "💼 专业背景", "callback_data": "adm:persona_materials:professional"}],
                [{"text": "🪪 个人简介", "callback_data": "adm:persona_materials:intro"}, {"text": "🖼️ 形象素材", "callback_data": "adm:persona_materials:visual"}],
                [{"text": "🗣️ 表达风格", "callback_data": "adm:persona_materials:voice"}, {"text": "🧷 连续性素材", "callback_data": "adm:persona_materials:continuity"}],
                [TGAdminMenuBuilder._back_row("adm:materials")[0]],
            ],
        }

    @staticmethod
    def scripts_menu() -> dict:
        return {
            "text": "📝 脚本管理\n请选择功能：",
            "reply_markup": [
                [{"text": "📚 查看脚本", "callback_data": "adm:scripts:list"}, {"text": "➕ 添加脚本", "callback_data": "adm:scripts:create"}],
                [{"text": "📦 按项目查看", "callback_data": "adm:scripts:projects"}, {"text": "🪜 按阶段查看", "callback_data": "adm:scripts:stages"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }

    @staticmethod
    def handover_menu() -> dict:
        return {
            "text": "🤝 接管管理\n请选择功能：",
            "reply_markup": [
                [{"text": "🧯 接管中列表", "callback_data": "adm:handover:active"}, {"text": "⏳ 待恢复列表", "callback_data": "adm:queue:pending_resume"}],
                [{"text": "📝 接管摘要", "callback_data": "adm:handover:summaries"}, {"text": "🔁 恢复建议", "callback_data": "adm:handover:resume"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }

    @staticmethod
    def queues_menu() -> dict:
        return {
            "text": "📬 队列中心\n请选择队列：",
            "reply_markup": [
                [{"text": "🔥 高意向队列", "callback_data": "adm:queue:high_intent"}, {"text": "⚠️ 紧急处理", "callback_data": "adm:queue:urgent_handover"}],
                [{"text": "⏳ 待恢复队列", "callback_data": "adm:queue:pending_resume"}, {"text": "📂 全部队列", "callback_data": "adm:queue:all"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }

    @staticmethod
    def receipts_menu() -> dict:
        return {
            "text": "🧾 回执审核\n请选择功能：",
            "reply_markup": [
                [{"text": "🕓 待审核回执", "callback_data": "adm:receipts:pending"}, {"text": "📦 项目分类回执", "callback_data": "adm:receipts:project"}],
                [{"text": "🏷️ 标签回执", "callback_data": "adm:receipts:tags"}, {"text": "🧯 覆盖记录", "callback_data": "adm:receipts:override"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }

    @staticmethod
    def settings_menu() -> dict:
        return {
            "text": "⚙️ 系统设置\n请选择功能：",
            "reply_markup": [
                [{"text": "🤖 AI 开关", "callback_data": "adm:settings:ai"}, {"text": "⏰ 工作时间", "callback_data": "adm:settings:hours"}],
                [{"text": "🧭 跟进设置", "callback_data": "adm:settings:followup"}, {"text": "📜 系统状态", "callback_data": "adm:settings:status"}],
                [TGAdminMenuBuilder._back_row()[0]],
            ],
        }


class TGAdminCallbackRouter:
    def __init__(self, admin_api_service: AdminAPIService, dashboard_service: DashboardService, tg_sender) -> None:
        self.admin_api_service = admin_api_service
        self.dashboard_service = dashboard_service
        self.tg_sender = tg_sender

    def _send_menu(self, admin_chat_id: int, menu: dict) -> None:
        self.tg_sender.send_admin_message(admin_chat_id, menu["text"], menu["reply_markup"])

    def _send_text(self, admin_chat_id: int, text: str) -> None:
        self.tg_sender.send_admin_text(chat_id=admin_chat_id, text=text)

    def _render_simple_list(self, title: str, rows: list[dict], keys: list[str]) -> str:
        if not rows:
            return f"{title}\n\n暂无数据。"
        lines = [title, ""]
        for idx, row in enumerate(rows[:20], start=1):
            parts = [f"{k}: {row.get(k)}" for k in keys if row.get(k) is not None]
            lines.append(f"{idx}. " + " | ".join(parts))
        return "\n".join(lines)

    def handle(self, admin_chat_id: int, callback_data: str, operator: str = "admin") -> None:
        parts = (callback_data or "").split(":")
        if not parts or parts[0] != "adm":
            self._send_text(admin_chat_id, "❌ 未知的管理员回调。")
            return
        action = parts[1] if len(parts) > 1 else "main"
        subaction = parts[2] if len(parts) > 2 else None

        if action == "main":
            self._send_menu(admin_chat_id, TGAdminMenuBuilder.main_menu()); return
        if action == "dashboard":
            summary = self.dashboard_service.get_summary(1)
            self._send_text(admin_chat_id, TGAdminFormatter.format_dashboard(summary)); return
        if action == "users":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.users_menu()); return
            if subaction == "list":
                data = self.dashboard_service.list_open_queue(1, None, 20)
                self._send_text(admin_chat_id, self._render_simple_list("👥 用户列表（开放队列近似）", data or [], ["conversation_id", "user_id", "queue_type", "priority"])); return
        if action == "classification":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.classification_menu()); return
            mapping = {"project": "📌 项目分类入口已接入，后续将补充用户级操作。", "tags": "🏷️ 标签分类入口已接入，后续将补充手动加减标签。", "ops": "🧭 运营分类入口已接入，后续将补充修改能力。", "locks": "🔒 锁定管理入口已接入，后续将补充锁定列表。"}
            self._send_text(admin_chat_id, mapping.get(subaction, "⚠️ 未知的分类操作。")); return
        if action == "projects":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.projects_menu()); return
            if subaction == "list":
                rows = self.admin_api_service.list_projects_for_business_account(1)
                self._send_text(admin_chat_id, self._render_simple_list("📦 项目列表", rows or [], ["id", "name", "status"])); return
            if subaction == "create":
                result = self.admin_api_service.create_project(1, "新项目", "待完善")
                self._send_text(admin_chat_id, f"➕ 添加项目结果：{result}"); return
            mapping = {"user_binding": "👤 用户项目绑定入口已接入。", "locks": "🔒 项目锁定入口已接入。"}
            self._send_text(admin_chat_id, mapping.get(subaction, "⚠️ 未知的项目操作。")); return
        if action == "materials":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.materials_menu()); return
            if subaction == "persona":
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.persona_materials_menu()); return
            if subaction == "project":
                rows = self.admin_api_service.list_project_materials(1)
                self._send_text(admin_chat_id, self._render_simple_list("📦 项目素材", rows or [], ["id", "material_type", "content_text", "media_url"])); return
            if subaction == "daily":
                rows = self.admin_api_service.list_persona_materials(1, "daily")
                self._send_text(admin_chat_id, self._render_simple_list("🌤️ 日常素材", rows or [], ["id", "material_type", "content_text", "media_url"])); return
            if subaction == "create":
                result = self.admin_api_service.create_persona_material(1, "daily", "示例日常话题")
                self._send_text(admin_chat_id, f"➕ 添加素材结果：{result}"); return
            self._send_text(admin_chat_id, "⚠️ 未知的素材操作。"); return
        if action == "persona_materials":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.persona_materials_menu()); return
            rows = self.admin_api_service.list_persona_materials(1, subaction)
            title_map = {"identity": "👤 基础身份素材", "professional": "💼 专业背景素材", "intro": "🪪 个人简介素材", "visual": "🖼️ 形象素材", "voice": "🗣️ 表达风格素材", "continuity": "🧷 连续性素材"}
            self._send_text(admin_chat_id, self._render_simple_list(title_map.get(subaction, "🧩 人设素材"), rows or [], ["id", "material_type", "content_text", "media_url"])); return
        if action == "scripts":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.scripts_menu()); return
            if subaction == "list":
                rows = self.admin_api_service.list_project_scripts(1)
                self._send_text(admin_chat_id, self._render_simple_list("📚 查看脚本", rows or [], ["id", "category", "content_text", "priority"])); return
            if subaction == "create":
                result = self.admin_api_service.create_project_script(1, "general", "示例脚本")
                self._send_text(admin_chat_id, f"➕ 添加脚本结果：{result}"); return
            if subaction == "projects":
                rows = self.admin_api_service.list_project_scripts(1)
                self._send_text(admin_chat_id, self._render_simple_list("📦 按项目查看脚本", rows or [], ["id", "category", "content_text", "priority"])); return
            mapping = {"stages": "🪜 按阶段查看脚本入口已接入。"}
            self._send_text(admin_chat_id, mapping.get(subaction, "⚠️ 未知的脚本操作。")); return
        if action == "handover":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.handover_menu()); return
            if subaction == "active":
                data = self.dashboard_service.list_open_queue(1, "urgent_handover", 20)
                self._send_text(admin_chat_id, self._render_simple_list("🧯 接管中/紧急处理列表", data or [], ["conversation_id", "user_id", "queue_type", "priority"])); return
            mapping = {"summaries": "📝 接管摘要入口已接入。", "resume": "🔁 恢复建议入口已接入。"}
            self._send_text(admin_chat_id, mapping.get(subaction, "⚠️ 未知的接管操作。")); return
        if action == "queues":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.queues_menu()); return
        if action == "queue":
            queue_type = subaction if subaction and subaction != "all" else None
            data = self.dashboard_service.list_open_queue(1, queue_type, 20)
            title_map = {None: "📂 全部队列", "high_intent": "🔥 高意向队列", "urgent_handover": "⚠️ 紧急处理队列", "pending_resume": "⏳ 待恢复队列"}
            self._send_text(admin_chat_id, self._render_simple_list(title_map.get(queue_type, "📬 队列"), data or [], ["conversation_id", "user_id", "queue_type", "priority"])); return
        if action == "receipts":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.receipts_menu()); return
            if subaction == "pending":
                rows = self.admin_api_service.receipt_repo.list_pending_receipts(1, 20) if hasattr(self.admin_api_service.receipt_repo, "list_pending_receipts") else []
                self._send_text(admin_chat_id, self._render_simple_list("🕓 待审核回执", rows or [], ["id", "receipt_type", "status", "user_id"])); return
            mapping = {"project": "📦 项目分类回执入口已接入。", "tags": "🏷️ 标签回执入口已接入。", "override": "🧯 覆盖记录入口已接入。"}
            self._send_text(admin_chat_id, mapping.get(subaction, "⚠️ 未知的回执操作。")); return
        if action == "settings":
            if subaction is None:
                self._send_menu(admin_chat_id, TGAdminMenuBuilder.settings_menu()); return
            mapping = {"ai": "🤖 AI 开关入口已接入。", "hours": "⏰ 工作时间入口已接入。", "followup": "🧭 跟进设置入口已接入。", "status": "📜 系统状态入口已接入。"}
            self._send_text(admin_chat_id, mapping.get(subaction, "⚠️ 未知的系统设置操作。")); return
        self._send_text(admin_chat_id, f"⚠️ Unsupported callback: {callback_data}")


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
# stage6 compatibility + advanced inbound chain
# =========================

class StrategyLearningRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_learning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def insert_record(self, *args: Any, **kwargs: Any) -> None:
        return None


class ProjectNurturePlanner(ProjectNurturePlanner):
    def plan(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        understanding = None
        stage_decision = None
        turn_decision = None
        project_decision = None
        user_state = None
        if len(args) == 5:
            understanding, stage_decision, turn_decision, project_decision, user_state = args
        elif len(args) == 4:
            turn_decision, understanding, stage_decision, user_state = args
            project_decision = kwargs.get('project_decision')
        else:
            understanding = kwargs.get('understanding')
            stage_decision = kwargs.get('stage_decision')
            turn_decision = kwargs.get('turn_decision')
            project_decision = kwargs.get('project_decision')
            user_state = kwargs.get('user_state')
        if turn_decision is None or understanding is None or stage_decision is None or user_state is None:
            return {
                'should_include_project_content': False,
                'nurture_angle': 'hold',
                'max_project_points': 0,
                'angle': 'presence_over_project',
                'visibility': 'none',
            }
        try:
            plan = super().plan(turn_decision, understanding, stage_decision, user_state)
        except TypeError:
            try:
                plan = super().plan(understanding, stage_decision, turn_decision, project_decision, user_state)
            except Exception:
                plan = {
                    'should_include_project_content': bool(getattr(turn_decision, 'should_push_project', False) and getattr(project_decision, 'project_id', None) is not None),
                    'nurture_angle': 'light_value_bridge',
                    'max_project_points': 1,
                }
        if not isinstance(plan, dict):
            plan = {'should_include_project_content': False, 'nurture_angle': 'hold', 'max_project_points': 0}
        should_push = bool(getattr(turn_decision, 'should_push_project', False) and getattr(project_decision, 'project_id', None) is not None)
        plan.setdefault('should_include_project_content', should_push)
        plan.setdefault('nurture_angle', 'light_value_bridge' if should_push else 'hold')
        plan.setdefault('max_project_points', 1 if should_push else 0)
        plan.setdefault('angle', plan.get('nurture_angle') or ('light_value_bridge' if should_push else 'presence_over_project'))
        visibility = 'light' if should_push else 'none'
        if getattr(turn_decision, 'project_window', 'hold') == 'hold' or getattr(turn_decision, 'pressure_guard', 'normal') == 'strict':
            visibility = 'none'
            plan['angle'] = 'presence_over_project'
        elif getattr(turn_decision, 'project_strategy', '') == 'clarify_then_pause':
            visibility = 'light'
            plan['angle'] = 'clear_fact_then_pause'
        plan['visibility'] = visibility
        return plan


def _orchestrator_handle_inbound_message_v2(self: Orchestrator, inbound_message: InboundMessage) -> None:
    business_account = self.business_account_repo.create_if_not_exists(
        inbound_message.tg_business_account_id,
        f"BusinessAccount-{inbound_message.tg_business_account_id}",
        None,
    )
    business_account_id = int(business_account['id'])
    self.bootstrap_repo.ensure_default_ai_control_settings(business_account_id)
    self.bootstrap_repo.ensure_default_persona_core(business_account_id)
    self.bootstrap_repo.ensure_default_tags(business_account_id)
    user_id = self.user_repo.get_or_create_user(inbound_message.tg_user_id, f"User-{inbound_message.tg_user_id}")
    conversation_id = self.conversation_repo.get_or_create_conversation(business_account_id, user_id)
    self.conversation_repo.save_message(conversation_id, 'user', inbound_message.message_type, inbound_message.text, inbound_message.raw_payload, inbound_message.media_url)
    context = self.conversation_repo.get_context(conversation_id)
    user_state = self.user_repo.get_user_state_snapshot(business_account_id, user_id)
    setattr(user_state, '_style_profile_data', getattr(user_state, '_style_profile_data', {}) or {})
    ai_allowed, ai_reason = self.ai_switch_engine.decide(business_account_id, conversation_id, user_state.ops_category, context.manual_takeover_status)
    if not ai_allowed:
        logger.info('AI reply skipped | conversation_id=%s | reason=%s', conversation_id, ai_reason)
        return

    latest_user_text = inbound_message.text or ''
    persona_profile = self.persona_profile_builder.build(business_account_id)
    persona_summary = self.persona_core.to_summary() + ' ' + self.persona_profile_builder.to_summary(persona_profile)
    recent_context = list(context.recent_messages)
    latest_handover_summary = None
    if self.handover_repo and context.manual_takeover_status == 'pending_resume':
        latest_handover_summary = self.handover_repo.get_latest_handover_summary_by_conversation(conversation_id)
        if latest_handover_summary:
            recent_context.append({
                'sender_type': 'system',
                'message_type': 'text',
                'content_text': f"[ResumeHint] theme={latest_handover_summary.get('theme_summary')}; user_state={latest_handover_summary.get('user_state_summary')}; resume={latest_handover_summary.get('resume_suggestion')}"
            })

    understanding = self.understanding_engine.analyze(latest_user_text, recent_context, self.persona_core.to_summary(), self._user_state_summary(user_state))
    stage_decision = self.stage_engine.decide(understanding, context.current_stage)
    trajectory = getattr(stage_decision, 'trajectory', 'steady')
    momentum = getattr(stage_decision, 'momentum', 'neutral')
    if hasattr(self, 'trajectory_engine') and self.trajectory_engine:
        try:
            trajectory, momentum = self.trajectory_engine.infer(understanding, user_state, context, [])
        except Exception as exc:
            logger.warning('trajectory inference failed: %s', exc)
    mode_decision = self.mode_router.decide(understanding, context.current_chat_mode)
    project_decision = self.project_classifier.classify(business_account_id, understanding, context, user_state, latest_user_text)
    intent_decision = self.intent_engine.detect(understanding, context, user_state, project_decision)
    segment_decision = self.project_segment_manager.decide(project_decision.project_id, understanding, intent_decision)
    escalation_decision = self.human_escalation_engine.decide(understanding, intent_decision)
    tag_decision = self.tagging_engine.decide(understanding, intent_decision, escalation_decision, user_state.tags)
    ops_decision = self.ops_category_manager.decide(user_state.ops_category, intent_decision, understanding)

    stored_profile = getattr(user_state, '_style_profile_data', {}) or {}
    rhythm_data = {'relationship_rhythm': 'steady', 'engagement_mode': 'maintain', 'trust_signal': 'neutral', 'recovery_window_hours': 24}
    maturity_data = {'maturity_state': 'emerging', 'maturity_reason': 'default', 'maturity_score': 0.5}
    if hasattr(self, 'rhythm_engine') and self.rhythm_engine:
        try:
            rhythm_data = self.rhythm_engine.decide(understanding, stage_decision, user_state, stored_profile=stored_profile)
        except Exception as exc:
            logger.warning('rhythm decision failed: %s', exc)
    if hasattr(self, 'maturity_engine') and self.maturity_engine:
        try:
            maturity_data = self.maturity_engine.assess(understanding, stage_decision, user_state, stored_profile=stored_profile, recent_learning=[])
        except Exception as exc:
            logger.warning('maturity assess failed: %s', exc)
    pressure_zone = getattr(stage_decision, 'pressure_zone', 'balanced')
    if hasattr(self, 'conflict_resolver') and self.conflict_resolver:
        try:
            resolved = self.conflict_resolver.resolve(understanding, stage_decision, rhythm_data, maturity_data, stored_profile=stored_profile)
            rhythm_data.update(resolved)
            pressure_zone = resolved.get('pressure_zone', pressure_zone)
        except Exception as exc:
            logger.warning('conflict resolve failed: %s', exc)
    stage_decision = StageDecision(
        stage=stage_decision.stage,
        changed=stage_decision.changed,
        reason=stage_decision.reason,
        confidence=getattr(stage_decision, 'confidence', 0.65),
        trajectory=trajectory,
        momentum=momentum,
        maturity_state=maturity_data.get('maturity_state', getattr(stage_decision, 'maturity_state', 'emerging')),
        maturity_reason=maturity_data.get('maturity_reason', getattr(stage_decision, 'maturity_reason', '')),
        maturity_score=maturity_data.get('maturity_score', getattr(stage_decision, 'maturity_score', 0.5)),
        pressure_zone=pressure_zone,
    )

    reply_plan = self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state)
    selected_content = self.content_selector.select(business_account_id, project_decision.project_id, mode_decision.chat_mode, reply_plan)

    turn_decision = None
    if hasattr(self, 'turn_decision_engine') and self.turn_decision_engine:
        try:
            turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
        except Exception as exc:
            logger.warning('turn decision failed: %s', exc)
    if turn_decision is None:
        turn_decision = TurnDecision(
            reply_goal=reply_plan.goal or 'rapport',
            should_push_project=reply_plan.should_continue_product,
            social_distance='balanced',
            reply_length='medium',
            ask_followup_question=not reply_plan.should_leave_space,
            exit_strategy='gentle_close' if reply_plan.should_leave_space else 'continue',
            self_disclosure_level='light' if reply_plan.should_self_share else 'none',
            reason=reply_plan.reason or 'reply_plan_fallback',
            should_reply=reply_plan.should_reply,
        )
    td = asdict(turn_decision)
    td['relationship_rhythm'] = rhythm_data.get('relationship_rhythm', td.get('relationship_rhythm', 'steady'))
    td['engagement_mode'] = rhythm_data.get('engagement_mode', td.get('engagement_mode', 'maintain'))
    td['recovery_window_hours'] = rhythm_data.get('recovery_window_hours', td.get('recovery_window_hours', 24))
    td['trust_signal'] = rhythm_data.get('trust_signal', td.get('trust_signal', 'neutral'))
    if hasattr(self, 'project_window_evaluator') and self.project_window_evaluator:
        try:
            td['project_window'] = self.project_window_evaluator.evaluate(understanding, stage_decision, turn_decision)
        except Exception as exc:
            logger.warning('project window eval failed: %s', exc)
    turn_decision = TurnDecision(**td)

    style_spec = StyleSpec()
    if hasattr(self, 'humanization_controller') and self.humanization_controller:
        try:
            style_spec = self.humanization_controller.build_style_spec(turn_decision, understanding, stage_decision, user_state, persona_profile)
        except Exception as exc:
            logger.warning('style spec build failed: %s', exc)

    try:
        selected_content['v5_project_nurture'] = self.project_nurture_planner.plan(understanding, stage_decision, turn_decision, project_decision, user_state)
    except Exception as exc:
        logger.warning('project nurture failed: %s', exc)
        selected_content['v5_project_nurture'] = {'should_include_project_content': False, 'angle': 'hold', 'visibility': 'none'}
    if hasattr(self, 'memory_priority_resolver') and self.memory_priority_resolver:
        try:
            selected_content['v5_memory_strategy'] = self.memory_priority_resolver.resolve(understanding, stage_decision, turn_preview=td, stored_profile=stored_profile)
        except Exception as exc:
            logger.warning('memory priority resolve failed: %s', exc)
    selected_content['v5_stage'] = asdict(stage_decision)
    selected_content['v5_turn_decision'] = asdict(turn_decision)
    selected_content['v5_style_spec'] = asdict(style_spec)
    selected_content['v5_segment'] = segment_decision

    understanding_payload = understanding.__dict__.copy()
    if latest_handover_summary:
        understanding_payload['resume_hint'] = latest_handover_summary.get('resume_suggestion')
    understanding_payload['trajectory'] = stage_decision.trajectory
    understanding_payload['momentum'] = stage_decision.momentum
    understanding_payload['maturity_state'] = stage_decision.maturity_state

    reply_plan_payload = {**reply_plan.__dict__, **asdict(turn_decision)}
    draft_reply = self.reply_style_engine.generate(
        latest_user_text, recent_context, persona_summary, self._user_state_summary(user_state),
        stage_decision.stage, mode_decision.chat_mode, understanding_payload,
        reply_plan_payload, selected_content,
    )
    final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
    if hasattr(self, 'maturity_polish_engine') and self.maturity_polish_engine:
        try:
            final_text = self.maturity_polish_engine.polish(final_text, understanding, style_spec, turn_decision)
        except Exception as exc:
            logger.warning('maturity polish failed: %s', exc)
    delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
    final_reply = FinalReply(text=final_text, delay_seconds=delay_seconds)
    if reply_plan.should_reply and final_reply.text.strip():
        self.sender_service.send_text_reply(conversation_id, final_reply.text, final_reply.delay_seconds)
        self.conversation_repo.save_message(conversation_id, 'ai', 'text', final_reply.text, {'selected_content': selected_content}, None)
        self.conversation_repo.set_last_ai_reply_at(conversation_id)

    self.conversation_repo.update_conversation_state(conversation_id, stage_decision.stage, mode_decision.chat_mode, understanding.current_mainline_should_continue)
    if project_decision.project_id is not None:
        self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
    self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
    if ops_decision['changed']:
        self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_decision['ops_category'], ops_decision['reason'], 'system')
    if hasattr(self, 'memory_writeback_engine') and self.memory_writeback_engine:
        try:
            self.memory_writeback_engine.write_after_turn(conversation_id, latest_user_text, final_reply.text, stage_decision, turn_decision, project_decision, intent_decision)
        except Exception as exc:
            logger.warning('memory writeback failed: %s', exc)
    if hasattr(self, 'handover_learning_engine') and self.handover_learning_engine:
        try:
            self.handover_learning_engine.learn_after_turn(business_account_id, user_id, conversation_id, context, turn_decision, stage_decision, understanding)
        except Exception as exc:
            logger.warning('handover learning failed: %s', exc)
    if escalation_decision.get('should_queue_admin'):
        queue_type = 'urgent_handover' if escalation_decision.get('notify_level') == 'urgent_takeover' else 'high_intent'
        priority_score = 95.0 if queue_type == 'urgent_handover' else (80.0 if escalation_decision.get('notify_level') == 'suggest_takeover' else 60.0)
        self.admin_queue_repo.upsert_queue_item(business_account_id, user_id, queue_type, priority_score, escalation_decision['reason'])
        self.receipt_repo.create_high_intent_receipt(
            business_account_id, user_id, 'High intent detected',
            {
                'notify_level': escalation_decision.get('notify_level'),
                'reason': escalation_decision.get('reason'),
                'project_id': project_decision.project_id,
                'segment_name': segment_decision.get('segment_name'),
                'intent_level': intent_decision.level,
                'intent_score': intent_decision.score,
            },
        )
        if self.admin_notifier:
            self.admin_notifier.notify_high_intent(business_account_id, user_id, conversation_id, escalation_decision.get('notify_level') or 'watch', escalation_decision.get('reason') or '')


Orchestrator.handle_inbound_message = _orchestrator_handle_inbound_message_v2

# ===== stage6 completion patch =====

class StyleDampingEngine:
    def stabilize(self, spec: StyleSpec, decision: TurnDecision, stored_profile: dict[str, Any] | None = None) -> StyleSpec:
        stored_profile = stored_profile or {}
        stable = spec
        style_profile = (stored_profile.get("style_profile") or decision.user_style_profile or "balanced").lower()
        if style_profile == "space_preferring":
            stable.sentence_density = "light"
            stable.softener_level = "soft"
            stable.initiative_level = "low"
        elif style_profile == "logic_first":
            stable.wording_texture = "clean"
            stable.directness = "clear"
        elif style_profile == "emotion_first":
            stable.warmth = "high"
            stable.softener_level = "high"
        return stable

_PrevAdaptiveStrategyOptimizer = AdaptiveStrategyOptimizer
class AdaptiveStrategyOptimizer(_PrevAdaptiveStrategyOptimizer):
    def infer_user_style(
        self,
        understanding: UnderstandingResult,
        user_state: UserStateSnapshot,
        stored_profile: dict[str, Any] | None = None,
        recent_learning: list[dict] | None = None,
    ) -> str:
        stored_profile = stored_profile or {}
        recent_learning = recent_learning or []
        style = _PrevAdaptiveStrategyOptimizer.infer_user_style(self, understanding, user_state)
        remembered = (stored_profile.get("style_profile") or "").strip()
        if remembered:
            style = remembered
        if any((r.get("user_style_profile") or "") == "space_preferring" for r in recent_learning[:3]):
            style = "space_preferring"
        if understanding.reluctance_type == "needs_space":
            style = "space_preferring"
        elif understanding.reluctance_type == "clarity_seek" and style == "balanced":
            style = "logic_first"
        return style

_PrevMemorySelector = MemorySelector
class MemorySelector(_PrevMemorySelector):
    def __init__(self, resolver: MemoryPriorityResolver | None = None, guard: ContinuityGuard | None = None, anchor_engine: AnchorMemoryEngine | None = None) -> None:
        self.resolver = resolver or MemoryPriorityResolver()
        self.guard = guard or ContinuityGuard()
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
        base = _PrevMemorySelector.select(self, memory_bundle, understanding, trajectory)
        stage_decision = stage_decision or StageDecision(stage="first_contact", changed=False, reason="fallback")
        resolved = self.resolver.resolve(understanding, stage_decision, stored_profile=stored_profile)
        priority = resolved.get("memory_priority", base.get("memory_focus", "relationship"))
        continuity_mode = resolved.get("continuity_mode", "implicit")
        selected_recent = self.guard.trim_recent(base.get("selected_recent_messages") or [], continuity_mode, priority)
        selected_summary = base.get("selected_recent_summary") if self.guard.should_surface_summary(understanding, priority) else ""
        anchor = self.anchor_engine.pick_anchor(selected_recent, selected_summary, continuity_mode, priority)
        return {
            **base,
            "selected_recent": selected_recent,
            "selected_summary": selected_summary,
            "memory_priority": priority,
            "continuity_mode": continuity_mode,
            "memory_focus": priority,
            "anchor_memory_hint": anchor,
        }

_PrevTurnDecisionEngine = TurnDecisionEngine
class TurnDecisionEngine(_PrevTurnDecisionEngine):
    def __init__(
        self,
        optimizer: AdaptiveStrategyOptimizer | None = None,
        rhythm_engine: RelationshipRhythmEngine | None = None,
        maturity_engine: RelationshipMaturityEngine | None = None,
        project_window_evaluator: ProjectWindowEvaluator | None = None,
        memory_resolver: MemoryPriorityResolver | None = None,
        conflict_resolver: RhythmConflictResolver | None = None,
    ) -> None:
        self.optimizer = optimizer or AdaptiveStrategyOptimizer()
        self.rhythm_engine = rhythm_engine or RelationshipRhythmEngine()
        self.maturity_engine = maturity_engine or RelationshipMaturityEngine()
        self.project_window_evaluator = project_window_evaluator or ProjectWindowEvaluator()
        self.memory_resolver = memory_resolver or MemoryPriorityResolver()
        self.conflict_resolver = conflict_resolver or RhythmConflictResolver()

    def decide(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> TurnDecision:
        decision = _PrevTurnDecisionEngine.decide(self, understanding, stage_decision, mode_decision, user_state)
        stored_profile = getattr(user_state, "_style_profile_data", None) or {}
        rhythm_data = self.rhythm_engine.decide(understanding, stage_decision, user_state, stored_profile=stored_profile)
        maturity_data = self.maturity_engine.assess(understanding, stage_decision, user_state, stored_profile=stored_profile)
        resolved = self.conflict_resolver.resolve(understanding, stage_decision, rhythm_data, maturity_data, stored_profile=stored_profile)
        data = {**asdict(decision)}
        data.update({
            "relationship_rhythm": resolved["relationship_rhythm"],
            "engagement_mode": resolved["engagement_mode"],
            "trust_signal": resolved["trust_signal"],
            "calibration_mode": resolved["calibration_mode"],
            "pressure_guard": resolved["pressure_guard"],
            "maturity_target": maturity_data.get("maturity_state", stage_decision.maturity_state),
            "maturity_score": float(maturity_data.get("maturity_score", stage_decision.maturity_score)),
        })
        mem = self.memory_resolver.resolve(understanding, stage_decision, stored_profile=stored_profile)
        data["memory_priority"] = mem.get("memory_priority", "relationship")
        data["continuity_mode"] = mem.get("continuity_mode", "implicit")
        data["memory_focus"] = data["memory_priority"]
        data["user_style_profile"] = self.optimizer.infer_user_style(understanding, user_state, stored_profile=stored_profile)
        td = TurnDecision(**data)
        td.project_window = self.project_window_evaluator.evaluate(understanding, stage_decision, td)
        td.window_confidence = 0.78 if td.project_window in ("open", "careful_open") else 0.52
        if understanding.reluctance_type == "needs_space":
            td.reply_length = "short"
            td.ask_followup_question = False
            td.should_push_project = False
            td.project_strategy = "hold_with_space"
        elif understanding.reluctance_type == "clarity_seek" and td.project_window in ("careful_open", "open"):
            td.project_strategy = "clarify_then_pause"
        return td

_PrevHumanizationController = HumanizationController
class HumanizationController(_PrevHumanizationController):
    def __init__(self, optimizer: AdaptiveStrategyOptimizer | None = None, damping_engine: StyleDampingEngine | None = None) -> None:
        self.optimizer = optimizer or AdaptiveStrategyOptimizer()
        self.damping_engine = damping_engine or StyleDampingEngine()

    def build_style_spec(
        self,
        decision: TurnDecision,
        understanding: UnderstandingResult,
        stage_decision: StageDecision,
        user_state: UserStateSnapshot,
        persona_profile: dict[str, Any] | None = None,
    ) -> StyleSpec:
        spec = _PrevHumanizationController.build_style_spec(self, decision, understanding, stage_decision, user_state, persona_profile)
        if decision.calibration_mode == "space_first":
            spec.pause_texture = "measured"
            spec.anchor_style = "implicit"
            spec.sentence_density = "light"
        elif decision.calibration_mode == "clear_progress":
            spec.pause_texture = "clean"
            spec.anchor_style = "fact"
        elif understanding.warmth_drift == "warming":
            spec.pause_texture = "soft"
            spec.anchor_style = "soft"
        stored_profile = getattr(user_state, "_style_profile_data", None) or {}
        return self.damping_engine.stabilize(spec, decision, stored_profile=stored_profile)

_PrevMemoryWritebackEngine = MemoryWritebackEngine
class MemoryWritebackEngine(_PrevMemoryWritebackEngine):
    def __init__(self, conversation_repo: ConversationRepository, strategy_learning_repo: StrategyLearningRepository | None = None) -> None:
        self.conversation_repo = conversation_repo
        self.strategy_learning_repo = strategy_learning_repo

    def _save_summary(self, conversation_id: int, summary: str) -> None:
        if hasattr(self.conversation_repo, 'save_summary'):
            self.conversation_repo.save_summary(conversation_id, summary)
            return
        db = getattr(self.conversation_repo, 'db', None)
        if db is None:
            return
        with db.transaction():
            with db.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversation_summaries (conversation_id, summary_text) VALUES (%s,%s)",
                    (conversation_id, summary),
                )

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
            f"stage={stage_decision.stage}; trajectory={stage_decision.trajectory}; goal={turn_decision.reply_goal}; "
            f"project_id={project_decision.project_id}; intent={intent_decision.level}; user='{user_text}'; ai='{reply_text}'"
        )
        self._save_summary(conversation_id, summary)

_PrevHandoverLearningEngine = HandoverLearningEngine
class HandoverLearningEngine(_PrevHandoverLearningEngine):
    def __init__(self, strategy_learning_repo: StrategyLearningRepository, abstraction_repo: Any | None = None) -> None:
        self.strategy_learning_repo = strategy_learning_repo
        self.abstraction_repo = abstraction_repo

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
        _PrevHandoverLearningEngine.learn_after_turn(self, business_account_id, user_id, conversation_id, context, turn_decision, stage_decision, understanding)
        if self.abstraction_repo:
            self.abstraction_repo.insert_abstraction(
                business_account_id,
                user_id,
                conversation_id,
                "calibration_trace",
                f"calibration={turn_decision.calibration_mode}; pressure_guard={turn_decision.pressure_guard}; anchor={turn_decision.anchor_memory_hint or 'none'}",
                understanding.reluctance_type or "none",
            )


def _orchestrator_handle_inbound_message_v6(self, inbound_message: InboundMessage) -> None:
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

    stored_profile = getattr(self, 'style_profile_repo', None)
    stored_profile = stored_profile.get_profile(business_account_id, user_id) if stored_profile and hasattr(stored_profile, 'get_profile') else {}
    recent_learning_repo = getattr(self, 'strategy_learning_repo', None)
    recent_learning = recent_learning_repo.list_recent_for_user(business_account_id, user_id, limit=5) if recent_learning_repo and hasattr(recent_learning_repo,'list_recent_for_user') else []
    setattr(user_state, '_style_profile_data', stored_profile or {})

    trajectory, momentum = self.trajectory_engine.infer(understanding, user_state, context, latest_summaries=[])
    stage_decision.trajectory = trajectory
    stage_decision.momentum = momentum
    maturity_data = self.maturity_engine.assess(understanding, stage_decision, user_state, stored_profile=stored_profile, recent_learning=recent_learning)
    stage_decision.maturity_state = maturity_data.get('maturity_state', stage_decision.maturity_state)
    stage_decision.maturity_reason = maturity_data.get('maturity_reason', stage_decision.maturity_reason)
    stage_decision.maturity_score = float(maturity_data.get('maturity_score', stage_decision.maturity_score))
    rhythm_data = self.rhythm_engine.decide(understanding, stage_decision, user_state, stored_profile=stored_profile)
    resolved = self.conflict_resolver.resolve(understanding, stage_decision, rhythm_data, maturity_data, stored_profile=stored_profile)
    stage_decision.pressure_zone = resolved.get('pressure_zone', stage_decision.pressure_zone)

    turn_decision = self.turn_decision_engine.decide(understanding, stage_decision, mode_decision, user_state)
    turn_decision.relationship_rhythm = resolved.get('relationship_rhythm', turn_decision.relationship_rhythm)
    turn_decision.engagement_mode = resolved.get('engagement_mode', turn_decision.engagement_mode)
    turn_decision.trust_signal = resolved.get('trust_signal', turn_decision.trust_signal)
    turn_decision.calibration_mode = resolved.get('calibration_mode', turn_decision.calibration_mode)
    turn_decision.pressure_guard = resolved.get('pressure_guard', turn_decision.pressure_guard)
    turn_decision.maturity_target = maturity_data.get('maturity_state', turn_decision.maturity_target)
    turn_decision.maturity_score = float(maturity_data.get('maturity_score', turn_decision.maturity_score))
    turn_decision.user_style_profile = self.strategy_optimizer.infer_user_style(understanding, user_state, stored_profile=stored_profile, recent_learning=recent_learning)

    memory_bundle = MemoryBundle(
        recent_messages=list(context.recent_messages or []),
        recent_summary=context.recent_summary or '',
        long_term_memory=context.long_term_memory or {},
        handover_summary=latest_handover_summary or {},
    )
    memory_selection = self.memory_selector.select(memory_bundle, understanding, trajectory, stored_profile=stored_profile, recent_learning=recent_learning, stage_decision=stage_decision)
    turn_decision.memory_priority = memory_selection.get('memory_priority', turn_decision.memory_priority)
    turn_decision.memory_focus = memory_selection.get('memory_focus', turn_decision.memory_focus)
    turn_decision.continuity_mode = memory_selection.get('continuity_mode', turn_decision.continuity_mode)
    turn_decision.anchor_memory_hint = memory_selection.get('anchor_memory_hint', turn_decision.anchor_memory_hint)
    turn_decision.project_window = self.project_window_evaluator.evaluate(understanding, stage_decision, turn_decision)

    project_nurture = self.project_nurture_planner.plan(understanding, stage_decision, turn_decision, project_decision, user_state)
    style_spec = self.humanization_controller.build_style_spec(turn_decision, understanding, stage_decision, user_state, persona_profile)

    selected_content = self.content_selector.select(business_account_id, project_decision.project_id, mode_decision.chat_mode, self.reply_planner.plan(understanding, stage_decision, mode_decision, user_state))
    selected_content['v5_memory'] = memory_selection
    selected_content['v5_project_nurture'] = project_nurture
    selected_content['v5_style_spec'] = asdict(style_spec)
    selected_content['v5_turn_decision'] = asdict(turn_decision)
    selected_content['v5_stage_decision'] = asdict(stage_decision)
    selected_content['v5_segment'] = segment_decision

    understanding_payload = understanding.__dict__.copy()
    if latest_handover_summary:
        understanding_payload['resume_hint'] = latest_handover_summary.get('resume_suggestion')
    understanding_payload['trajectory'] = trajectory
    understanding_payload['momentum'] = momentum
    understanding_payload['pressure_zone'] = stage_decision.pressure_zone

    draft_reply = self.reply_style_engine.generate(
        latest_user_text,
        memory_selection.get('selected_recent') or recent_context,
        persona_summary,
        self._user_state_summary(user_state),
        stage_decision.stage,
        mode_decision.chat_mode,
        understanding_payload,
        asdict(turn_decision),
        selected_content,
    )
    final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
    final_text = self.maturity_polish_engine.polish(final_text, understanding, style_spec, turn_decision)
    delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
    final_reply = FinalReply(text=final_text, delay_seconds=delay_seconds)

    if turn_decision.should_reply and final_reply.text.strip():
        self.sender_service.send_text_reply(conversation_id, final_reply.text, final_reply.delay_seconds)
        self.conversation_repo.save_message(conversation_id, 'ai', 'text', final_reply.text, {'selected_content': selected_content}, None)
        self.conversation_repo.set_last_ai_reply_at(conversation_id)

    self.conversation_repo.update_conversation_state(conversation_id, stage_decision.stage, mode_decision.chat_mode, understanding.current_mainline_should_continue)
    if project_decision.project_id is not None:
        self.user_repo.update_project_state(business_account_id, user_id, project_decision.project_id, project_decision.reason)
    self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)
    if ops_decision['changed']:
        self.user_repo.set_ops_category_manual(business_account_id, user_id, ops_decision['ops_category'], ops_decision['reason'], 'system')
    if escalation_decision.get('should_queue_admin'):
        queue_type = 'urgent_handover' if escalation_decision.get('notify_level') == 'urgent_takeover' else 'high_intent'
        priority_score = 95.0 if queue_type == 'urgent_handover' else (80.0 if escalation_decision.get('notify_level') == 'suggest_takeover' else 60.0)
        self.admin_queue_repo.upsert_queue_item(business_account_id, user_id, queue_type, priority_score, escalation_decision['reason'])
        self.receipt_repo.create_high_intent_receipt(
            business_account_id, user_id, 'High intent detected',
            {
                'notify_level': escalation_decision.get('notify_level'),
                'reason': escalation_decision.get('reason'),
                'project_id': project_decision.project_id,
                'segment_name': segment_decision.get('segment_name'),
                'intent_level': intent_decision.level,
                'intent_score': intent_decision.score,
            },
        )
        if self.admin_notifier:
            self.admin_notifier.notify_high_intent(business_account_id, user_id, conversation_id, escalation_decision.get('notify_level') or 'watch', escalation_decision.get('reason') or '')

    try:
        self.memory_writeback_engine.write_after_turn(conversation_id, latest_user_text, final_reply.text, stage_decision, turn_decision, project_decision, intent_decision)
    except Exception:
        logger.exception('memory_writeback failed | conversation_id=%s', conversation_id)
    try:
        self.handover_learning_engine.learn_after_turn(business_account_id, user_id, conversation_id, context, turn_decision, stage_decision, understanding)
    except Exception:
        logger.exception('handover_learning failed | conversation_id=%s', conversation_id)

Orchestrator.handle_inbound_message = _orchestrator_handle_inbound_message_v6


def build_app_components(settings: Settings) -> dict[str, Any]:
    db = Database(settings.database_url)
    db.connect()
    initialize_database(db)
    tg_client = TelegramBotAPIClient(bot_token=settings.tg_bot_token, db=db, admin_chat_ids=settings.admin_chat_ids)
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
    strategy_learning_repo = StrategyLearningRepository(db)
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
    trajectory_engine = RelationshipTrajectoryEngine()
    rhythm_engine = RelationshipRhythmEngine()
    maturity_engine = RelationshipMaturityEngine()
    conflict_resolver = RhythmConflictResolver()
    memory_priority_resolver = MemoryPriorityResolver()
    strategy_optimizer = AdaptiveStrategyOptimizer()
    project_window_evaluator = ProjectWindowEvaluator()
    memory_selector = MemorySelector(memory_priority_resolver, ContinuityGuard(), AnchorMemoryEngine())
    turn_decision_engine = TurnDecisionEngine(strategy_optimizer, rhythm_engine, maturity_engine, project_window_evaluator, memory_priority_resolver, conflict_resolver)
    humanization_controller = HumanizationController(strategy_optimizer, StyleDampingEngine())
    maturity_polish_engine = MaturityPolishEngine()
    memory_writeback_engine = MemoryWritebackEngine(conversation_repo, strategy_learning_repo)
    handover_learning_engine = HandoverLearningEngine(strategy_learning_repo)
    handover_manager = HandoverManager(handover_repo, user_control_repo, conversation_repo, admin_queue_repo)
    handover_summary_builder = HandoverSummaryBuilder(handover_repo, conversation_repo, llm_service)
    resume_chat_manager = ResumeChatManager(handover_repo, conversation_repo, user_control_repo)
    customer_actions = CustomerActions(user_control_repo, user_repo, handover_manager, handover_summary_builder, resume_chat_manager, handover_repo, conversation_repo)
    admin_api_service = AdminAPIService(user_repo, receipt_repo, handover_repo, conversation_repo, user_control_repo, admin_queue_repo, customer_actions, resume_chat_manager, project_repo, material_repo, script_repo)
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
    orchestrator.trajectory_engine = trajectory_engine
    orchestrator.rhythm_engine = rhythm_engine
    orchestrator.maturity_engine = maturity_engine
    orchestrator.conflict_resolver = conflict_resolver
    orchestrator.memory_priority_resolver = memory_priority_resolver
    orchestrator.strategy_optimizer = strategy_optimizer
    orchestrator.project_window_evaluator = project_window_evaluator
    orchestrator.memory_selector = memory_selector
    orchestrator.turn_decision_engine = turn_decision_engine
    orchestrator.humanization_controller = humanization_controller
    orchestrator.maturity_polish_engine = maturity_polish_engine
    orchestrator.memory_writeback_engine = memory_writeback_engine
    orchestrator.handover_learning_engine = handover_learning_engine
    orchestrator.strategy_learning_repo = strategy_learning_repo
    gateway = TelegramBusinessGateway(settings, orchestrator.handle_inbound_message)
    return {
        'db': db,
        'llm_service': llm_service,
        'sender_service': sender_service,
        'admin_notifier': admin_notifier,
        'gateway': gateway,
        'orchestrator': orchestrator,
        'handover_manager': handover_manager,
        'handover_summary_builder': handover_summary_builder,
        'resume_chat_manager': resume_chat_manager,
        'customer_actions': customer_actions,
        'admin_api_service': admin_api_service,
        'dashboard_service': dashboard_service,
        'tg_admin_callback_router': tg_admin_callback_router,
        'tg_admin_handlers': tg_admin_handlers,
        'tg_client': tg_client,
    }

if __name__ == "__main__":
    main()
