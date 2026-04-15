from __future__ import annotations

import json
import copy
import re
import logging
import os
import sys
import time
import hmac
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
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
class FollowupDecision:
    should_schedule: bool
    followup_kind: str = "none"
    scheduled_for: datetime | None = None
    reason: str = ""
    text: str = ""
    pacing_mode: str = "normal"
    snooze_until: datetime | None = None


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
        CREATE TABLE IF NOT EXISTS daily_materials (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            category TEXT,
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
        CREATE TABLE IF NOT EXISTS material_packages (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            project_id BIGINT NULL REFERENCES projects(id) ON DELETE CASCADE,
            package_scope TEXT NOT NULL,
            package_name TEXT NOT NULL,
            description TEXT,
            package_kind TEXT,
            scene_tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            allow_partial BOOLEAN NOT NULL DEFAULT TRUE,
            language_code TEXT,
            priority INTEGER NOT NULL DEFAULT 100,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_material_packages_scope_project
            ON material_packages (business_account_id, package_scope, project_id, priority)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS material_package_items (
            id BIGSERIAL PRIMARY KEY,
            package_id BIGINT NOT NULL REFERENCES material_packages(id) ON DELETE CASCADE,
            item_type TEXT NOT NULL,
            content_text TEXT,
            media_url TEXT,
            caption_text TEXT,
            item_meta_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            sort_order INTEGER NOT NULL DEFAULT 0,
            is_required BOOLEAN NOT NULL DEFAULT FALSE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_material_package_items_package_id
            ON material_package_items (package_id, sort_order, id)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS material_package_buttons (
            id BIGSERIAL PRIMARY KEY,
            package_id BIGINT NOT NULL REFERENCES material_packages(id) ON DELETE CASCADE,
            button_text TEXT NOT NULL,
            button_type TEXT NOT NULL DEFAULT 'url',
            button_value TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_material_package_buttons_package_id
            ON material_package_buttons (package_id, sort_order, id)
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
        ''',
        '''
        CREATE TABLE IF NOT EXISTS webhook_event_locks (
            id BIGSERIAL PRIMARY KEY,
            event_key VARCHAR(255) NOT NULL UNIQUE,
            event_type VARCHAR(50) NOT NULL,
            business_account_id BIGINT NULL,
            business_connection_id VARCHAR(128) NULL,
            telegram_chat_id BIGINT NULL,
            telegram_user_id BIGINT NULL,
            telegram_message_id BIGINT NULL,
            telegram_update_id BIGINT NULL,
            is_edited BOOLEAN NOT NULL DEFAULT FALSE,
            status VARCHAR(20) NOT NULL DEFAULT 'processing',
            error_message TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processed_at TIMESTAMPTZ NULL
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_webhook_event_locks_created_at
            ON webhook_event_locks (created_at DESC)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_webhook_event_locks_account_msg
            ON webhook_event_locks (business_account_id, telegram_chat_id, telegram_message_id)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS admin_sessions (
            admin_chat_id BIGINT PRIMARY KEY,
            business_account_id BIGINT NULL,
            state VARCHAR(50) NOT NULL DEFAULT 'idle',
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NULL
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_admin_sessions_state
            ON admin_sessions (state)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_admin_sessions_updated_at
            ON admin_sessions (updated_at DESC)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS outbound_jobs (
            id BIGSERIAL PRIMARY KEY,
            job_type VARCHAR(32) NOT NULL,
            business_account_id BIGINT NOT NULL,
            target_chat_id BIGINT NOT NULL,
            target_user_id BIGINT NULL,
            dedup_key VARCHAR(255) NULL UNIQUE,
            source_inbound_message_id BIGINT NULL,
            payload_json JSONB NOT NULL,
            send_after TIMESTAMPTZ NOT NULL,
            priority SMALLINT NOT NULL DEFAULT 50,
            status VARCHAR(20) NOT NULL DEFAULT 'queued',
            retry_count INT NOT NULL DEFAULT 0,
            max_retry_count INT NOT NULL DEFAULT 3,
            last_error TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            locked_at TIMESTAMPTZ NULL,
            sent_at TIMESTAMPTZ NULL
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_outbound_jobs_due
            ON outbound_jobs (status, send_after)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_outbound_jobs_account_due
            ON outbound_jobs (business_account_id, status, send_after)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_outbound_jobs_source_msg
            ON outbound_jobs (source_inbound_message_id)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS admin_action_receipts (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NULL,
            admin_chat_id BIGINT NOT NULL,
            action_type VARCHAR(50) NOT NULL,
            target_user_id BIGINT NULL,
            target_project_id BIGINT NULL,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            result_status VARCHAR(20) NOT NULL DEFAULT 'success',
            message_text TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_admin_action_receipts_admin_chat_id
            ON admin_action_receipts (admin_chat_id, created_at DESC)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_admin_action_receipts_action_type
            ON admin_action_receipts (action_type, created_at DESC)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS business_message_edits (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NULL,
            telegram_chat_id BIGINT NOT NULL,
            telegram_user_id BIGINT NULL,
            telegram_message_id BIGINT NOT NULL,
            linked_inbound_message_id BIGINT NULL,
            edited_text TEXT NULL,
            message_type VARCHAR(30) NULL,
            raw_payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_business_message_edits_chat_msg
            ON business_message_edits (telegram_chat_id, telegram_message_id)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_business_message_edits_created_at
            ON business_message_edits (created_at DESC)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS business_runtime_policies (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL UNIQUE REFERENCES business_accounts(id) ON DELETE CASCADE,
            timezone_name TEXT NOT NULL DEFAULT 'UTC',
            active_weekdays_json JSONB NOT NULL DEFAULT '[1,2,3,4,5,6,7]'::jsonb,
            reply_window_start_hour SMALLINT NOT NULL DEFAULT 9,
            reply_window_end_hour SMALLINT NOT NULL DEFAULT 21,
            inbound_ai_policy TEXT NOT NULL DEFAULT 'always',
            proactive_followup_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            proactive_followup_policy TEXT NOT NULL DEFAULT 'workhours_only',
            min_followup_delay_minutes INT NOT NULL DEFAULT 720,
            max_followup_delay_minutes INT NOT NULL DEFAULT 2160,
            busy_snooze_hours INT NOT NULL DEFAULT 18,
            stop_snooze_hours INT NOT NULL DEFAULT 72,
            daily_max_followups_per_user INT NOT NULL DEFAULT 1,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_business_runtime_policies_timezone
            ON business_runtime_policies (timezone_name)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS conversation_pacing_state (
            id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
            pacing_mode TEXT NOT NULL DEFAULT 'normal',
            boundary_signal TEXT,
            pressure_level TEXT,
            next_followup_after TIMESTAMPTZ,
            followup_snooze_until TIMESTAMPTZ,
            last_boundary_at TIMESTAMPTZ,
            last_user_message_at TIMESTAMPTZ,
            last_ai_message_at TIMESTAMPTZ,
            notes_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_conversation_pacing_state_followup
            ON conversation_pacing_state (next_followup_after, followup_snooze_until)
        ''',
        '''
        CREATE TABLE IF NOT EXISTS proactive_followup_plans (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            followup_kind TEXT NOT NULL,
            reason_text TEXT,
            status TEXT NOT NULL DEFAULT 'scheduled',
            scheduled_for TIMESTAMPTZ NOT NULL,
            dedup_key VARCHAR(255) NULL UNIQUE,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_message_id BIGINT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            sent_at TIMESTAMPTZ NULL,
            cancelled_at TIMESTAMPTZ NULL,
            last_error TEXT NULL
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_proactive_followup_plans_due
            ON proactive_followup_plans (status, scheduled_for)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_proactive_followup_plans_conversation
            ON proactive_followup_plans (conversation_id, status, scheduled_for)
        ''',
        '''
        ALTER TABLE conversation_summaries
            ADD COLUMN IF NOT EXISTS summary_json JSONB NOT NULL DEFAULT '{}'::jsonb
        ''',
        '''
        ALTER TABLE conversation_summaries
            ADD COLUMN IF NOT EXISTS summary_kind TEXT NOT NULL DEFAULT 'rolling'
        ''',
        '''
        ALTER TABLE conversation_summaries
            ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'system'
        ''',
        '''
        ALTER TABLE conversation_summaries
            ADD COLUMN IF NOT EXISTS message_count INT NOT NULL DEFAULT 0
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_conversation_summaries_conversation_id_created_at
            ON conversation_summaries (conversation_id, created_at DESC)
        ''',
        '''
        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS reply_language_mode VARCHAR(20) NOT NULL DEFAULT 'fixed_en'
        ''',
        '''
        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS preferred_language VARCHAR(10) NOT NULL DEFAULT 'en'
        ''',
        '''
        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS detected_language VARCHAR(10) NULL
        ''',
        '''
        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS language_confidence NUMERIC(4,3) NULL
        ''',
        '''
        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS language_source VARCHAR(20) NOT NULL DEFAULT 'system_default'
        ''',
        '''
        UPDATE users
        SET
            reply_language_mode = COALESCE(reply_language_mode, 'fixed_en'),
            preferred_language = COALESCE(preferred_language, 'en'),
            language_source = COALESCE(language_source, 'system_default')
        WHERE
            reply_language_mode IS NULL
            OR preferred_language IS NULL
            OR language_source IS NULL
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

    def get_by_id(self, business_account_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM business_accounts WHERE id=%s LIMIT 1", (business_account_id,))
            return cur.fetchone()

    def get_by_tg_business_account_id(self, tg_business_account_id: str) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM business_accounts WHERE tg_business_account_id=%s LIMIT 1", (tg_business_account_id,))
            return cur.fetchone()

    def get_default_account(self) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM business_accounts ORDER BY id ASC LIMIT 1")
            return cur.fetchone()

    def get_default_account_id(self) -> int | None:
        row = self.get_default_account()
        return None if not row else int(row["id"])

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

    def resolve_from_connection(self, tg_business_account_id: str | None, display_name: str | None = None, username: str | None = None) -> dict | None:
        if tg_business_account_id:
            return self.create_if_not_exists(
                tg_business_account_id,
                display_name or f"BusinessAccount-{tg_business_account_id}",
                username,
            )
        return self.get_default_account()


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

    def save_message(self, conversation_id: int, sender_type: str, message_type: str, text: str | None, raw_payload: dict | None = None, media_url: str | None = None) -> int:
        raw_payload_json = json.dumps(raw_payload or {}, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (conversation_id, sender_type, message_type, content_text, media_url, raw_payload_json)
                    VALUES (%s,%s,%s,%s,%s,%s::jsonb)
                    RETURNING id
                    """,
                    (conversation_id, sender_type, message_type, text, media_url, raw_payload_json),
                )
                message_id = int(cur.fetchone()["id"])
                cur.execute("UPDATE conversations SET last_message_at=NOW(), updated_at=NOW() WHERE id=%s", (conversation_id,))
                return message_id

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
                """
                SELECT summary_text, summary_json, summary_kind, source, message_count, created_at
                FROM conversation_summaries
                WHERE conversation_id=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
            summary_row = cur.fetchone()
        memory_payload = {}
        if summary_row and isinstance(summary_row.get("summary_json"), dict):
            memory_payload = summary_row.get("summary_json") or {}
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
            long_term_memory=memory_payload,
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

    def get_latest_summary_row(self, conversation_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT id, summary_text, summary_json, summary_kind, source, message_count, created_at
                FROM conversation_summaries
                WHERE conversation_id=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
            return cur.fetchone()

    def get_messages_for_summary(self, conversation_id: int, limit: int = 30) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT id, sender_type, message_type, content_text, media_url, created_at
                FROM messages
                WHERE conversation_id=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (conversation_id, limit),
            )
            rows = cur.fetchall()
        return list(reversed(rows))

    def save_conversation_summary(self, conversation_id: int, summary_text: str, summary_json: dict[str, Any] | None = None,
                                  summary_kind: str = "rolling", source: str = "ai", message_count: int = 0) -> int:
        summary_payload = json.dumps(summary_json or {}, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_summaries (
                        conversation_id, summary_text, summary_json, summary_kind, source, message_count, created_at
                    )
                    VALUES (%s,%s,%s::jsonb,%s,%s,%s,NOW())
                    RETURNING id
                    """,
                    (conversation_id, summary_text, summary_payload, summary_kind, source, int(message_count or 0)),
                )
                row = cur.fetchone()
                return int(row["id"])

    def count_messages_since_latest_summary(self, conversation_id: int) -> int:
        latest = self.get_latest_summary_row(conversation_id)
        with self.db.cursor() as cur:
            if latest and latest.get("created_at") is not None:
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id=%s AND created_at>=%s",
                    (conversation_id, latest["created_at"]),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id=%s",
                    (conversation_id,),
                )
            row = cur.fetchone()
        return int((row or {}).get("cnt") or 0)


class UserRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_or_create_user(self, tg_user_id: str, display_name: str | None = None, username: str | None = None, language_code: str | None = None) -> int:
        with self.db.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE tg_user_id=%s LIMIT 1", (tg_user_id,))
            row = cur.fetchone()
            if row:
                user_id = int(row["id"])
                with self.db.transaction():
                    with self.db.cursor() as cur2:
                        cur2.execute(
                            "UPDATE users SET display_name=COALESCE(%s, display_name), username=COALESCE(%s, username), language_code=COALESCE(%s, language_code), last_seen_at=NOW(), updated_at=NOW() WHERE id=%s",
                            (display_name, username, language_code, user_id),
                        )
                return user_id
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (tg_user_id, username, display_name, language_code, reply_language_mode, preferred_language, language_source, first_seen_at, last_seen_at, is_blocked, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,'fixed_en','en','system_default',NOW(),NOW(),FALSE,NOW(),NOW()) RETURNING id
                    """,
                    (tg_user_id, username, display_name, language_code),
                )
                return int(cur.fetchone()["id"])

    def get_user_profile(self, business_account_id: int, user_id: int) -> dict:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT id, tg_user_id, username, display_name, language_code, reply_language_mode, preferred_language, detected_language, language_confidence, language_source, first_seen_at, last_seen_at, is_blocked FROM users WHERE id=%s LIMIT 1",
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
                SELECT ups.project_id, ups.status, ups.source, ups.is_locked, ups.reason_text, ups.confidence,
                       ups.candidate_projects_json, ups.updated_at, p.name AS project_name
                FROM user_project_state ups LEFT JOIN projects p ON ups.project_id=p.id
                WHERE ups.business_account_id=%s AND ups.user_id=%s LIMIT 1
                """,
                (business_account_id, user_id),
            )
            project_row = cur.fetchone()
            cur.execute(
                """
                SELECT upss.project_segment_id, upss.reason_text, upss.source, upss.updated_at, ps.name AS segment_name
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

    def update_project_state(self, business_account_id: int, user_id: int, project_id: int | None, reason: str,
                             candidate_projects: list[dict[str, Any]] | None = None, confidence: float | None = None,
                             source: str = "ai", status: str = "classified", updated_by: str = "system") -> dict[str, Any]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT project_id, is_locked, source, status
                FROM user_project_state
                WHERE business_account_id=%s AND user_id=%s
                LIMIT 1
                """,
                (business_account_id, user_id),
            )
            existing = cur.fetchone()
        if existing and existing.get("is_locked") and (existing.get("source") or "") == "manual":
            return {"updated": False, "skipped_locked": True, "project_id": existing.get("project_id")}
        candidate_projects_json = json.dumps(candidate_projects or [], ensure_ascii=False)
        final_confidence = float(confidence if confidence is not None else (0.7 if project_id is not None else 0.0))
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_project_state (
                        business_account_id, user_id, project_id, candidate_projects_json,
                        source, reason_text, confidence, status, is_locked, updated_by, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s,FALSE,%s,NOW(),NOW())
                    ON CONFLICT (business_account_id, user_id)
                    DO UPDATE SET project_id=EXCLUDED.project_id,
                                  candidate_projects_json=EXCLUDED.candidate_projects_json,
                                  source=EXCLUDED.source,
                                  reason_text=EXCLUDED.reason_text,
                                  confidence=EXCLUDED.confidence,
                                  status=EXCLUDED.status,
                                  updated_by=EXCLUDED.updated_by,
                                  updated_at=NOW()
                    """,
                    (business_account_id, user_id, project_id, candidate_projects_json, source, reason, final_confidence, status, updated_by),
                )
        return {"updated": True, "skipped_locked": False, "project_id": project_id}

    def update_project_segment_state(self, business_account_id: int, user_id: int, project_id: int | None,
                                     project_segment_id: int | None, reason_text: str, source: str = "ai",
                                     updated_by: str = "system") -> dict[str, Any]:
        if project_id is None:
            project_segment_id = None
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_project_segment_state (
                        business_account_id, user_id, project_id, project_segment_id,
                        source, reason_text, is_locked, updated_by, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,FALSE,%s,NOW(),NOW())
                    ON CONFLICT (business_account_id, user_id)
                    DO UPDATE SET project_id=EXCLUDED.project_id,
                                  project_segment_id=EXCLUDED.project_segment_id,
                                  source=EXCLUDED.source,
                                  reason_text=EXCLUDED.reason_text,
                                  updated_by=EXCLUDED.updated_by,
                                  updated_at=NOW()
                    """,
                    (business_account_id, user_id, project_id, project_segment_id, source, reason_text, updated_by),
                )
        return {"updated": True, "project_segment_id": project_segment_id}

    def apply_tag_decision(self, business_account_id: int, user_id: int, tag_decision: TagDecision) -> dict[str, Any]:
        added: list[str] = []
        removed: list[str] = []
        skipped_locked: list[str] = []
        missing: list[str] = []
        with self.db.transaction():
            with self.db.cursor() as cur:
                for tag_name in tag_decision.add_tags:
                    cur.execute(
                        """
                        SELECT t.id, ut.is_locked, ut.is_active
                        FROM tags t
                        LEFT JOIN user_tags ut
                          ON ut.business_account_id=%s AND ut.user_id=%s AND ut.tag_id=t.id
                        WHERE t.business_account_id=%s AND t.name=%s
                        LIMIT 1
                        """,
                        (business_account_id, user_id, business_account_id, tag_name),
                    )
                    tag_row = cur.fetchone()
                    if not tag_row:
                        missing.append(tag_name)
                        continue
                    if tag_row.get("is_locked") and tag_row.get("is_active"):
                        skipped_locked.append(tag_name)
                        continue
                    cur.execute(
                        """
                        INSERT INTO user_tags (
                            business_account_id, user_id, tag_id, source, reason_text,
                            confidence, is_active, is_locked, expires_at, created_by, created_at, updated_at
                        ) VALUES (%s,%s,%s,'ai',%s,0.7,TRUE,FALSE,NULL,'system',NOW(),NOW())
                        ON CONFLICT (business_account_id, user_id, tag_id)
                        DO UPDATE SET source=EXCLUDED.source,
                                      reason_text=EXCLUDED.reason_text,
                                      confidence=EXCLUDED.confidence,
                                      is_active=TRUE,
                                      updated_at=NOW()
                        """,
                        (business_account_id, user_id, tag_row["id"], tag_decision.reason),
                    )
                    if tag_name not in added:
                        added.append(tag_name)
                for tag_name in tag_decision.remove_tags:
                    cur.execute(
                        """
                        SELECT t.id, ut.is_locked, ut.is_active
                        FROM tags t
                        LEFT JOIN user_tags ut
                          ON ut.business_account_id=%s AND ut.user_id=%s AND ut.tag_id=t.id
                        WHERE t.business_account_id=%s AND t.name=%s
                        LIMIT 1
                        """,
                        (business_account_id, user_id, business_account_id, tag_name),
                    )
                    tag_row = cur.fetchone()
                    if not tag_row:
                        missing.append(tag_name)
                        continue
                    if tag_row.get("is_locked"):
                        skipped_locked.append(tag_name)
                        continue
                    if not tag_row.get("is_active"):
                        continue
                    cur.execute(
                        "UPDATE user_tags SET is_active=FALSE, updated_at=NOW() WHERE business_account_id=%s AND user_id=%s AND tag_id=%s",
                        (business_account_id, user_id, tag_row["id"]),
                    )
                    if tag_name not in removed:
                        removed.append(tag_name)
        return {
            "added": added,
            "removed": removed,
            "skipped_locked": list(dict.fromkeys(skipped_locked)),
            "missing": list(dict.fromkeys(missing)),
            "reason": tag_decision.reason,
        }


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

    @staticmethod
    def _normalize_scene_tags(row: dict[str, Any] | None) -> dict[str, Any]:
        normalized = dict(row or {})
        tags = normalized.get("scene_tags_json")
        if tags is None:
            normalized["scene_tags_json"] = []
        elif isinstance(tags, str):
            normalized["scene_tags_json"] = [tags]
        else:
            normalized["scene_tags_json"] = list(tags)
        return normalized

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
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

    def get_daily_materials(self, business_account_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM daily_materials WHERE business_account_id=%s AND is_active=TRUE ORDER BY priority ASC, id ASC",
                (business_account_id,),
            )
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

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
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

    def list_material_packages(self, business_account_id: int, package_scope: str | None = None, project_id: int | None = None) -> list[dict]:
        conditions = ["business_account_id=%s", "is_active=TRUE"]
        params: list[Any] = [business_account_id]
        if package_scope:
            conditions.append("package_scope=%s")
            params.append(package_scope)
        if package_scope == "project":
            conditions.append("project_id=%s")
            params.append(project_id or -1)
        elif project_id is not None:
            conditions.append("(project_id IS NULL OR project_id=%s)")
            params.append(project_id)
        query = f"SELECT * FROM material_packages WHERE {' AND '.join(conditions)} ORDER BY priority ASC, id ASC"
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

    def get_material_package_items(self, package_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM material_package_items
                WHERE package_id=%s AND is_active=TRUE
                ORDER BY sort_order ASC, id ASC
                """,
                (package_id,),
            )
            return cur.fetchall()

    def get_material_package_buttons(self, package_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM material_package_buttons
                WHERE package_id=%s AND is_active=TRUE
                ORDER BY sort_order ASC, id ASC
                """,
                (package_id,),
            )
            return cur.fetchall()

    def build_material_package(self, package_row: dict[str, Any]) -> dict[str, Any]:
        package_payload = self._normalize_scene_tags(package_row)
        package_payload["items"] = self.get_material_package_items(int(package_row["id"]))
        package_payload["buttons"] = self.get_material_package_buttons(int(package_row["id"]))
        return package_payload

    def get_material_library_overview(self, business_account_id: int, project_id: int | None = None) -> dict[str, Any]:
        daily_count = len(self.get_daily_materials(business_account_id))
        persona_count = len(self.get_persona_materials(business_account_id))
        project_material_count = len(self.get_project_materials(project_id)) if project_id else 0
        persona_packages = len(self.list_material_packages(business_account_id, "persona"))
        daily_packages = len(self.list_material_packages(business_account_id, "daily"))
        project_packages = len(self.list_material_packages(business_account_id, "project", project_id)) if project_id else 0
        return {
            "persona_material_count": persona_count,
            "daily_material_count": daily_count,
            "project_material_count": project_material_count,
            "persona_package_count": persona_packages,
            "daily_package_count": daily_packages,
            "project_package_count": project_packages,
        }

    def list_persona_materials_admin(self, business_account_id: int, include_inactive: bool = True) -> list[dict]:
        query = "SELECT * FROM persona_materials WHERE business_account_id=%s"
        params: list[Any] = [business_account_id]
        if not include_inactive:
            query += " AND is_active=TRUE"
        query += " ORDER BY priority ASC, id ASC"
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

    def list_daily_materials_admin(self, business_account_id: int, include_inactive: bool = True) -> list[dict]:
        query = "SELECT * FROM daily_materials WHERE business_account_id=%s"
        params: list[Any] = [business_account_id]
        if not include_inactive:
            query += " AND is_active=TRUE"
        query += " ORDER BY priority ASC, id ASC"
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

    def list_project_materials_admin(self, project_id: int, include_inactive: bool = True) -> list[dict]:
        query = "SELECT * FROM project_materials WHERE project_id=%s"
        params: list[Any] = [project_id]
        if not include_inactive:
            query += " AND is_active=TRUE"
        query += " ORDER BY priority ASC, id ASC"
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

    def create_persona_material(self, business_account_id: int, material_type: str, content_text: str,
                                media_url: str | None = None, scene_tags: list[str] | None = None,
                                priority: int = 100, is_active: bool = True) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO persona_materials (
                        business_account_id, material_type, content_text, media_url, scene_tags_json, priority, is_active, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,NOW(),NOW()) RETURNING *
                    """,
                    (business_account_id, material_type, content_text, media_url, json.dumps(scene_tags or [], ensure_ascii=False), priority, is_active),
                )
                return self._normalize_scene_tags(cur.fetchone())

    def create_daily_material(self, business_account_id: int, category: str, content_text: str,
                              media_url: str | None = None, scene_tags: list[str] | None = None,
                              priority: int = 100, is_active: bool = True) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO daily_materials (
                        business_account_id, category, content_text, media_url, scene_tags_json, priority, is_active, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,NOW(),NOW()) RETURNING *
                    """,
                    (business_account_id, category, content_text, media_url, json.dumps(scene_tags or [], ensure_ascii=False), priority, is_active),
                )
                return self._normalize_scene_tags(cur.fetchone())

    def create_project_material(self, project_id: int, material_type: str, content_text: str,
                                media_url: str | None = None, scene_tags: list[str] | None = None,
                                priority: int = 100, is_active: bool = True) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO project_materials (
                        project_id, material_type, content_text, media_url, scene_tags_json, priority, is_active, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,NOW(),NOW()) RETURNING *
                    """,
                    (project_id, material_type, content_text, media_url, json.dumps(scene_tags or [], ensure_ascii=False), priority, is_active),
                )
                return self._normalize_scene_tags(cur.fetchone())

    def update_material_active(self, table_name: str, material_id: int, is_active: bool) -> bool:
        if table_name not in {"persona_materials", "daily_materials", "project_materials"}:
            return False
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(f"UPDATE {table_name} SET is_active=%s, updated_at=NOW() WHERE id=%s RETURNING id", (is_active, material_id))
                return cur.fetchone() is not None


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

    def list_projects_admin(self, business_account_id: int, include_inactive: bool = True) -> list[dict]:
        query = "SELECT * FROM projects WHERE business_account_id=%s"
        params: list[Any] = [business_account_id]
        if not include_inactive:
            query += " AND is_active=TRUE"
        query += " ORDER BY id ASC"
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return cur.fetchall()

    def get_project_by_id(self, project_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM projects WHERE id=%s LIMIT 1", (project_id,))
            return cur.fetchone()

    def create_project(self, business_account_id: int, name: str, description: str | None = None, is_active: bool = True) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projects (business_account_id, name, description, is_active, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,NOW(),NOW()) RETURNING *
                    """,
                    (business_account_id, name.strip(), description, is_active),
                )
                return cur.fetchone()

    def update_project_active(self, project_id: int, is_active: bool) -> bool:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE projects SET is_active=%s, updated_at=NOW() WHERE id=%s RETURNING id", (is_active, project_id))
                return cur.fetchone() is not None

    def get_project_segments(self, project_id: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM project_segments WHERE project_id=%s AND is_active=TRUE ORDER BY sort_order ASC, id ASC",
                (project_id,),
            )
            return cur.fetchall()

    def list_project_segments_admin(self, project_id: int, include_inactive: bool = True) -> list[dict]:
        query = "SELECT * FROM project_segments WHERE project_id=%s"
        params: list[Any] = [project_id]
        if not include_inactive:
            query += " AND is_active=TRUE"
        query += " ORDER BY sort_order ASC, id ASC"
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return cur.fetchall()

    def create_project_segment(self, project_id: int, name: str, description: str | None = None,
                               sort_order: int = 0, is_active: bool = True) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO project_segments (project_id, name, description, sort_order, is_active, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,%s,NOW(),NOW()) RETURNING *
                    """,
                    (project_id, name.strip(), description, sort_order, is_active),
                )
                return cur.fetchone()

    def update_project_segment_active(self, segment_id: int, is_active: bool) -> bool:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE project_segments SET is_active=%s, updated_at=NOW() WHERE id=%s RETURNING id", (is_active, segment_id))
                return cur.fetchone() is not None


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

    def list_project_scripts_admin(self, business_account_id: int, project_id: int | None = None, include_inactive: bool = True) -> list[dict]:
        conditions = ["p.business_account_id=%s"]
        params: list[Any] = [business_account_id]
        if project_id is not None:
            conditions.append("ps.project_id=%s")
            params.append(project_id)
        if not include_inactive:
            conditions.append("ps.is_active=TRUE")
        query = f"""
            SELECT ps.*, p.name AS project_name
            FROM project_scripts ps
            JOIN projects p ON p.id=ps.project_id
            WHERE {' AND '.join(conditions)}
            ORDER BY ps.priority ASC, ps.id ASC
        """
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return cur.fetchall()

    def create_project_script(self, project_id: int, category: str, content_text: str,
                              priority: int = 100, is_active: bool = True) -> dict:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO project_scripts (project_id, category, content_text, priority, is_active, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,%s,NOW(),NOW()) RETURNING *
                    """,
                    (project_id, category.strip(), content_text, priority, is_active),
                )
                return cur.fetchone()

    def update_project_script_active(self, script_id: int, is_active: bool) -> bool:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE project_scripts SET is_active=%s, updated_at=NOW() WHERE id=%s RETURNING id", (is_active, script_id))
                return cur.fetchone() is not None


class ReceiptRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create_high_intent_receipt(self, business_account_id: int, user_id: int, title: str, content_json: dict) -> None:
        self.create_system_receipt(business_account_id, user_id, 'high_intent', title, title, content_json)

    def create_system_receipt(self, business_account_id: int, user_id: int, receipt_type: str, title: str,
                              content_text: str, content_json: dict[str, Any]) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO receipts (
                        business_account_id, user_id, receipt_type, title, content_text, content_json, status, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,'pending',NOW())
                    """,
                    (business_account_id, user_id, receipt_type, title, content_text, content_json),
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


class WebhookEventLockRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def acquire(self, event_key: str, event_type: str, business_account_id: int | None, business_connection_id: str | None,
                telegram_chat_id: int | None, telegram_user_id: int | None, telegram_message_id: int | None,
                telegram_update_id: int | None, is_edited: bool) -> bool:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO webhook_event_locks (
                        event_key, event_type, business_account_id, business_connection_id,
                        telegram_chat_id, telegram_user_id, telegram_message_id, telegram_update_id,
                        is_edited, status, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'processing',NOW())
                    ON CONFLICT (event_key) DO NOTHING
                    RETURNING id
                    """,
                    (event_key, event_type, business_account_id, business_connection_id, telegram_chat_id, telegram_user_id, telegram_message_id, telegram_update_id, is_edited),
                )
                return cur.fetchone() is not None

    def mark_processed(self, event_key: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE webhook_event_locks SET status='processed', processed_at=NOW(), error_message=NULL WHERE event_key=%s", (event_key,))

    def mark_failed(self, event_key: str, error_message: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE webhook_event_locks SET status='failed', processed_at=NOW(), error_message=%s WHERE event_key=%s", (error_message[:1000], event_key))

    def record_duplicate_skip(self, event_key: str) -> None:
        logger.info("Duplicate webhook event skipped | event_key=%s", event_key)


class AdminSessionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_session(self, admin_chat_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM admin_sessions WHERE admin_chat_id=%s LIMIT 1", (admin_chat_id,))
            return cur.fetchone()

    def set_session(self, admin_chat_id: int, state: str, payload: dict[str, Any] | None = None, business_account_id: int | None = None, expires_at: datetime | None = None) -> None:
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO admin_sessions (admin_chat_id, business_account_id, state, payload_json, created_at, updated_at, expires_at)
                    VALUES (%s,%s,%s,%s::jsonb,NOW(),NOW(),%s)
                    ON CONFLICT (admin_chat_id)
                    DO UPDATE SET business_account_id=EXCLUDED.business_account_id,
                                  state=EXCLUDED.state,
                                  payload_json=EXCLUDED.payload_json,
                                  updated_at=NOW(),
                                  expires_at=EXCLUDED.expires_at
                    """,
                    (admin_chat_id, business_account_id, state, payload_json, expires_at),
                )

    def clear_session(self, admin_chat_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("DELETE FROM admin_sessions WHERE admin_chat_id=%s", (admin_chat_id,))

    def touch_session(self, admin_chat_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE admin_sessions SET updated_at=NOW() WHERE admin_chat_id=%s", (admin_chat_id,))


class OutboundJobRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def enqueue(self, job_type: str, business_account_id: int, target_chat_id: int, target_user_id: int | None,
                payload: dict[str, Any], send_after: datetime, dedup_key: str | None = None,
                source_inbound_message_id: int | None = None, priority: int = 50) -> int | None:
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO outbound_jobs (
                        job_type, business_account_id, target_chat_id, target_user_id,
                        dedup_key, source_inbound_message_id, payload_json, send_after,
                        priority, status, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,'queued',NOW(),NOW())
                    ON CONFLICT (dedup_key) DO NOTHING
                    RETURNING id
                    """,
                    (job_type, business_account_id, target_chat_id, target_user_id, dedup_key, source_inbound_message_id, payload_json, send_after, priority),
                )
                row = cur.fetchone()
                return None if row is None else int(row['id'])

    def fetch_due(self, limit: int = 20) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM outbound_jobs
                WHERE status='queued' AND send_after<=NOW()
                ORDER BY priority ASC, send_after ASC, id ASC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()

    def mark_sending(self, job_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE outbound_jobs SET status='sending', locked_at=NOW(), updated_at=NOW() WHERE id=%s", (job_id,))

    def mark_sent(self, job_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE outbound_jobs SET status='sent', sent_at=NOW(), updated_at=NOW(), last_error=NULL WHERE id=%s", (job_id,))

    def mark_failed(self, job_id: int, error: str, retryable: bool = True) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                if retryable:
                    cur.execute(
                        """
                        UPDATE outbound_jobs
                        SET status=CASE WHEN retry_count + 1 >= max_retry_count THEN 'failed' ELSE 'queued' END,
                            retry_count=retry_count + 1,
                            last_error=%s,
                            send_after=CASE WHEN retry_count + 1 >= max_retry_count THEN send_after ELSE NOW() + INTERVAL '10 seconds' END,
                            updated_at=NOW()
                        WHERE id=%s
                        """,
                        (error[:1000], job_id),
                    )
                else:
                    cur.execute("UPDATE outbound_jobs SET status='failed', retry_count=retry_count + 1, last_error=%s, updated_at=NOW() WHERE id=%s", (error[:1000], job_id))


class BusinessMessageEditRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record_edit(self, business_account_id: int | None, telegram_chat_id: int, telegram_user_id: int | None,
                    telegram_message_id: int, edited_text: str | None, message_type: str | None,
                    raw_payload: dict[str, Any], linked_inbound_message_id: int | None = None) -> None:
        raw_payload_json = json.dumps(raw_payload, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO business_message_edits (
                        business_account_id, telegram_chat_id, telegram_user_id, telegram_message_id,
                        linked_inbound_message_id, edited_text, message_type, raw_payload, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,NOW())
                    """,
                    (business_account_id, telegram_chat_id, telegram_user_id, telegram_message_id, linked_inbound_message_id, edited_text, message_type, raw_payload_json),
                )


class RuntimePolicyRepository:
    def __init__(self, db: Database, default_timezone: str = 'UTC') -> None:
        self.db = db
        self.default_timezone = default_timezone or 'UTC'

    def ensure_default_policy(self, business_account_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO business_runtime_policies (
                        business_account_id, timezone_name, active_weekdays_json,
                        reply_window_start_hour, reply_window_end_hour,
                        inbound_ai_policy, proactive_followup_enabled, proactive_followup_policy,
                        min_followup_delay_minutes, max_followup_delay_minutes,
                        busy_snooze_hours, stop_snooze_hours, daily_max_followups_per_user,
                        created_at, updated_at
                    )
                    VALUES (%s,%s,'[1,2,3,4,5,6,7]'::jsonb,9,21,'always',TRUE,'workhours_only',720,2160,18,72,1,NOW(),NOW())
                    ON CONFLICT (business_account_id) DO NOTHING
                    """,
                    (business_account_id, self.default_timezone),
                )

    def get_policy(self, business_account_id: int) -> dict[str, Any]:
        self.ensure_default_policy(business_account_id)
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM business_runtime_policies WHERE business_account_id=%s LIMIT 1", (business_account_id,))
            row = cur.fetchone() or {}
        policy = dict(row or {})
        weekdays = policy.get('active_weekdays_json')
        if isinstance(weekdays, str):
            try:
                weekdays = json.loads(weekdays)
            except Exception:
                weekdays = [1, 2, 3, 4, 5, 6, 7]
        policy['active_weekdays_json'] = list(weekdays or [1, 2, 3, 4, 5, 6, 7])
        policy['timezone_name'] = str(policy.get('timezone_name') or self.default_timezone)
        return policy


class PacingStateRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_state(self, conversation_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM conversation_pacing_state WHERE conversation_id=%s LIMIT 1", (conversation_id,))
            row = cur.fetchone()
        return None if not row else dict(row)

    def upsert_state(self, conversation_id: int, pacing_mode: str, boundary_signal: str | None,
                     next_followup_after: datetime | None, followup_snooze_until: datetime | None,
                     pressure_level: str | None = None, notes: dict[str, Any] | None = None,
                     touch_user: bool = False, touch_ai: bool = False) -> None:
        notes_json = json.dumps(notes or {}, ensure_ascii=False)
        boundary_present = boundary_signal is not None
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_pacing_state (
                        conversation_id, pacing_mode, boundary_signal, pressure_level,
                        next_followup_after, followup_snooze_until, last_boundary_at,
                        last_user_message_at, last_ai_message_at, notes_json, created_at, updated_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,
                        CASE WHEN %s::boolean THEN NOW() ELSE NULL END,
                        CASE WHEN %s::boolean THEN NOW() ELSE NULL END,
                        CASE WHEN %s::boolean THEN NOW() ELSE NULL END,
                        %s::jsonb,NOW(),NOW()
                    )
                    ON CONFLICT (conversation_id)
                    DO UPDATE SET pacing_mode=EXCLUDED.pacing_mode,
                                  boundary_signal=EXCLUDED.boundary_signal,
                                  pressure_level=EXCLUDED.pressure_level,
                                  next_followup_after=EXCLUDED.next_followup_after,
                                  followup_snooze_until=EXCLUDED.followup_snooze_until,
                                  last_boundary_at=CASE WHEN EXCLUDED.boundary_signal IS NULL THEN conversation_pacing_state.last_boundary_at ELSE NOW() END,
                                  last_user_message_at=CASE WHEN %s::boolean THEN NOW() ELSE conversation_pacing_state.last_user_message_at END,
                                  last_ai_message_at=CASE WHEN %s::boolean THEN NOW() ELSE conversation_pacing_state.last_ai_message_at END,
                                  notes_json=EXCLUDED.notes_json,
                                  updated_at=NOW()
                    """,
                    (
                        conversation_id, pacing_mode, boundary_signal, pressure_level,
                        next_followup_after, followup_snooze_until, boundary_present,
                        touch_user, touch_ai, notes_json, touch_user, touch_ai,
                    ),
                )


class ProactiveFollowupPlanRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def cancel_open_plans(self, conversation_id: int, reason_text: str = 'cancelled') -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE proactive_followup_plans
                    SET status='cancelled', cancelled_at=NOW(), updated_at=NOW(), reason_text=%s
                    WHERE conversation_id=%s AND status IN ('scheduled','processing')
                    """,
                    (reason_text[:500], conversation_id),
                )

    def schedule_plan(self, business_account_id: int, conversation_id: int, user_id: int,
                      followup_kind: str, scheduled_for: datetime, payload: dict[str, Any],
                      reason_text: str, source_message_id: int | None = None) -> int:
        dedup_key = f"followup:{conversation_id}:{followup_kind}"
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO proactive_followup_plans (
                        business_account_id, conversation_id, user_id, followup_kind,
                        reason_text, status, scheduled_for, dedup_key, payload_json,
                        source_message_id, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s,'scheduled',%s,%s,%s::jsonb,%s,NOW(),NOW())
                    ON CONFLICT (dedup_key)
                    DO UPDATE SET scheduled_for=EXCLUDED.scheduled_for,
                                  payload_json=EXCLUDED.payload_json,
                                  reason_text=EXCLUDED.reason_text,
                                  status='scheduled',
                                  cancelled_at=NULL,
                                  sent_at=NULL,
                                  updated_at=NOW(),
                                  last_error=NULL
                    RETURNING id
                    """,
                    (business_account_id, conversation_id, user_id, followup_kind, reason_text[:500], scheduled_for, dedup_key, payload_json, source_message_id),
                )
                return int(cur.fetchone()['id'])

    def fetch_due(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM proactive_followup_plans
                WHERE status='scheduled' AND scheduled_for <= NOW()
                ORDER BY scheduled_for ASC, id ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def mark_processing(self, plan_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE proactive_followup_plans SET status='processing', updated_at=NOW() WHERE id=%s", (plan_id,))

    def mark_sent(self, plan_id: int) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE proactive_followup_plans SET status='sent', sent_at=NOW(), updated_at=NOW(), last_error=NULL WHERE id=%s", (plan_id,))

    def reschedule(self, plan_id: int, next_time: datetime, reason_text: str) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "UPDATE proactive_followup_plans SET status='scheduled', scheduled_for=%s, reason_text=%s, updated_at=NOW() WHERE id=%s",
                    (next_time, reason_text[:500], plan_id),
                )

    def mark_failed(self, plan_id: int, error_text: str, retryable: bool = True) -> None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                if retryable:
                    cur.execute(
                        "UPDATE proactive_followup_plans SET status='scheduled', scheduled_for=NOW() + INTERVAL '30 minutes', last_error=%s, updated_at=NOW() WHERE id=%s",
                        (error_text[:1000], plan_id),
                    )
                else:
                    cur.execute(
                        "UPDATE proactive_followup_plans SET status='failed', last_error=%s, updated_at=NOW() WHERE id=%s",
                        (error_text[:1000], plan_id),
                    )

    def count_open(self, business_account_id: int) -> int:
        with self.db.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM proactive_followup_plans WHERE business_account_id=%s AND status IN ('scheduled','processing')", (business_account_id,))
            row = cur.fetchone()
        return int((row or {}).get('cnt') or 0)

    def get_next_for_conversation(self, conversation_id: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM proactive_followup_plans
                WHERE conversation_id=%s AND status IN ('scheduled','processing')
                ORDER BY scheduled_for ASC LIMIT 1
                """,
                (conversation_id,),
            )
            row = cur.fetchone()
        return None if not row else dict(row)


class AdminActionReceiptRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create_receipt(self, business_account_id: int | None, admin_chat_id: int, action_type: str, message_text: str,
                       target_user_id: int | None = None, target_project_id: int | None = None,
                       payload: dict[str, Any] | None = None, result_status: str = 'success') -> None:
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO admin_action_receipts (
                        business_account_id, admin_chat_id, action_type, target_user_id, target_project_id,
                        payload_json, result_status, message_text, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s,NOW())
                    """,
                    (business_account_id, admin_chat_id, action_type, target_user_id, target_project_id, payload_json, result_status, message_text[:1000]),
                )

    def list_recent_by_user(self, business_account_id: int, user_id: int, limit: int = 10) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM admin_action_receipts
                WHERE business_account_id=%s AND target_user_id=%s
                ORDER BY created_at DESC
                LIMIT %s
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

    def summarize_conversation(self, prompt: str) -> dict[str, Any]:
        return self._parse_json_result(
            self._call_text_model(prompt, 0.2, 900),
            {
                "summary_text": "",
                "key_facts": [],
                "user_preferences": [],
                "boundaries": [],
                "project_signals": [],
                "followup_strategy": "",
                "relationship_notes": "",
                "tags_to_watch": [],
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


def build_understanding_prompt(latest_user_message: str, recent_context: list[dict[str, Any]], persona_core_summary: str,
                               user_state_summary: str, long_term_memory_summary: str = "") -> str:
    recent_messages = [
        {"sender_type": m.get("sender_type"), "message_type": m.get("message_type"), "content_text": m.get("content_text")}
        for m in recent_context[-10:]
    ]
    return f"""
You are a conversation understanding engine for a Telegram business account AI chat assistant.
Return JSON only.
Persona core summary: {persona_core_summary}
User state summary: {user_state_summary}
Long-term memory summary: {long_term_memory_summary}
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


def resolve_user_reply_language(user_profile: dict[str, Any]) -> str:
    user_row = (user_profile or {}).get("user") or {}
    mode = (user_row.get("reply_language_mode") or "fixed_en").lower()
    if mode == "fixed_en":
        return "English"
    preferred = (user_row.get("preferred_language") or "en").lower()
    return {"en": "English", "de": "German", "fr": "French"}.get(preferred, "English")


def build_reply_prompt(latest_user_message: str, recent_context: list[dict[str, Any]], persona_summary: str,
                       user_state_summary: str, stage: str, chat_mode: str, understanding: dict[str, Any],
                       reply_plan: dict[str, Any], selected_content: dict[str, Any],
                       memory_summary: dict[str, Any] | None = None, reply_language: str = "English") -> str:
    recent_messages = [{"sender_type": m.get("sender_type"), "content_text": m.get("content_text")} for m in recent_context[-10:]]
    memory_payload = memory_summary or {}
    selected_payload = dict(selected_content or {})
    selected_content_summary = str(selected_payload.get("prompt_summary") or "")
    selected_prompt_packages = selected_payload.get("prompt_packages") or {}
    compact_selected = {
        "project_scripts": [
            {"id": item.get("id"), "category": item.get("category"), "text": (item.get("content_text") or "")[:160]}
            for item in (selected_payload.get("project_scripts") or [])[:3]
        ],
        "project_materials": _summarize_material_items(selected_payload.get("project_materials") or [], 3),
        "persona_materials": _summarize_material_items(selected_payload.get("persona_materials") or [], 3),
        "daily_materials": _summarize_material_items(selected_payload.get("daily_materials") or [], 3),
        "material_packages": selected_prompt_packages,
    }
    return f"""
You are the live chat brain for a Telegram business account.
Write exactly one natural reply message in plain text.
Persona summary: {persona_summary}
User state summary: {user_state_summary}
Stage: {stage}
Chat mode: {chat_mode}
Long-term memory: {json.dumps(memory_payload, ensure_ascii=False)}
Understanding: {json.dumps(understanding, ensure_ascii=False)}
Reply plan: {json.dumps(reply_plan, ensure_ascii=False)}
Selected content summary: {selected_content_summary}
Selected content detail: {json.dumps(compact_selected, ensure_ascii=False)}
Recent messages: {json.dumps(recent_messages, ensure_ascii=False)}
Latest user message: {latest_user_message}
Rules:
- Reply language for the current phase: {reply_language}.
- For the current phase, user-facing replies must stay in natural English only.
- Do not switch to another language unless a later architecture upgrade enables it.
- Use long-term memory to stay consistent with user preferences, boundaries, and prior progress.
- Use the selected scripts/materials/packages as optional support, not as a rigid script.
- Material packages may contain multiple images, videos, text snippets, or buttons; keep your reply naturally aligned with the most relevant package when useful.
- If mode is pause, respect the boundary and do not continue product pushing.
- If mode is emotional, acknowledge emotion first.
- If mode is product or high_intent, answer the user's product question directly and stay on the product mainline.
- Avoid sounding robotic, over-perfect, or overly salesy.
""".strip()


def build_conversation_summary_prompt(conversation_context: dict[str, Any], recent_messages: list[dict[str, Any]],
                                      latest_user_message: str, understanding: dict[str, Any],
                                      state_snapshot: dict[str, Any], latest_reply: str | None = None) -> str:
    normalized_messages = [
        {
            "sender_type": item.get("sender_type"),
            "message_type": item.get("message_type"),
            "content_text": item.get("content_text"),
            "created_at": str(item.get("created_at")),
        }
        for item in recent_messages[-16:]
    ]
    return f"""
You are summarizing a Telegram business chat so the assistant can keep long-term memory.
Return JSON only.
Conversation context: {json.dumps(conversation_context, ensure_ascii=False)}
Recent messages: {json.dumps(normalized_messages, ensure_ascii=False)}
Latest user message: {latest_user_message}
Latest assistant reply: {latest_reply or ""}
Understanding: {json.dumps(understanding, ensure_ascii=False)}
State snapshot: {json.dumps(state_snapshot, ensure_ascii=False)}
Schema:
{{
  "summary_text": "short rolling summary in one paragraph",
  "key_facts": ["important stable facts"],
  "user_preferences": ["stable preferences or language/social style"],
  "boundaries": ["busy/later/avoid pressure etc"],
  "project_signals": ["current project interest/progress"],
  "followup_strategy": "how the assistant should continue next time",
  "relationship_notes": "trust/rhythm/pace notes",
  "tags_to_watch": ["followup_worthy", "recently_busy"]
}}
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

    def _prepare_reply_markup(self, reply_markup: Any) -> dict[str, Any] | None:
        if not reply_markup:
            return None
        if isinstance(reply_markup, dict) and reply_markup.get("inline_keyboard"):
            return reply_markup
        return {"inline_keyboard": reply_markup}

    @staticmethod
    def _normalize_button_value(button_type: str, button_value: str) -> str:
        value = str(button_value or "").strip()
        if not value:
            return value
        normalized_type = str(button_type or "url").strip().lower()
        if value.startswith("http://") or value.startswith("https://") or value.startswith("tg://"):
            return value
        if value.startswith("t.me/"):
            return f"https://{value}"
        if value.startswith("@"):
            return f"https://t.me/{value[1:]}"
        if normalized_type in {"telegram_user", "telegram_account", "telegram_group", "telegram_channel", "telegram", "tg"}:
            return f"https://t.me/{value.lstrip('/')}"
        if normalized_type in {"url", "link", "external"} and "/" not in value and "." not in value:
            return f"https://t.me/{value}"
        return value

    def build_inline_keyboard(self, buttons: list[dict[str, Any]] | None, row_width: int = 2) -> dict[str, Any] | None:
        if not buttons:
            return None
        keyboard: list[list[dict[str, Any]]] = []
        row: list[dict[str, Any]] = []
        for button in buttons:
            text_value = str(button.get("text") or button.get("button_text") or "").strip()
            value = self._normalize_button_value(
                str(button.get("type") or button.get("button_type") or "url"),
                str(button.get("value") or button.get("button_value") or ""),
            )
            if not text_value or not value:
                continue
            row.append({"text": text_value, "url": value})
            if len(row) >= max(1, row_width):
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
        if not keyboard:
            return None
        return {"inline_keyboard": keyboard}

    def send_admin_text(self, chat_id: int, text: str) -> None:
        self._call_api("sendMessage", {"chat_id": chat_id, "text": text})

    def send_admin_message(self, chat_id: int, text: str, reply_markup=None) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        prepared = self._prepare_reply_markup(reply_markup)
        if prepared:
            payload["reply_markup"] = prepared
        self._call_api("sendMessage", payload)

    def send_text(self, conversation_id: int, text: str, reply_markup=None) -> None:
        delivery = self._get_delivery_context_for_conversation(conversation_id)
        payload: dict[str, Any] = {
            "chat_id": delivery["chat_id"],
            "text": text,
            "business_connection_id": delivery.get("business_connection_id"),
        }
        prepared = self._prepare_reply_markup(reply_markup)
        if prepared:
            payload["reply_markup"] = prepared
        self._call_api("sendMessage", payload)

    def send_photo(self, conversation_id: int, media_url: str, caption: str | None = None, reply_markup=None) -> None:
        delivery = self._get_delivery_context_for_conversation(conversation_id)
        payload: dict[str, Any] = {
            "chat_id": delivery["chat_id"],
            "photo": media_url,
            "caption": caption,
            "business_connection_id": delivery.get("business_connection_id"),
        }
        prepared = self._prepare_reply_markup(reply_markup)
        if prepared:
            payload["reply_markup"] = prepared
        self._call_api("sendPhoto", payload)

    def send_video(self, conversation_id: int, media_url: str, caption: str | None = None, reply_markup=None) -> None:
        delivery = self._get_delivery_context_for_conversation(conversation_id)
        payload: dict[str, Any] = {
            "chat_id": delivery["chat_id"],
            "video": media_url,
            "caption": caption,
            "business_connection_id": delivery.get("business_connection_id"),
        }
        prepared = self._prepare_reply_markup(reply_markup)
        if prepared:
            payload["reply_markup"] = prepared
        self._call_api("sendVideo", payload)

    def send_media_group(self, conversation_id: int, media_items: list[dict[str, Any]]) -> None:
        delivery = self._get_delivery_context_for_conversation(conversation_id)
        media_payload: list[dict[str, Any]] = []
        for idx, item in enumerate(media_items[:10]):
            media_type = str(item.get("type") or item.get("media_type") or "photo").lower()
            if media_type not in {"photo", "video"}:
                media_type = "photo"
            entry: dict[str, Any] = {
                "type": media_type,
                "media": item.get("media_url") or item.get("media"),
            }
            caption_value = str(item.get("caption") or "").strip()
            if idx == 0 and caption_value:
                entry["caption"] = caption_value
            media_payload.append(entry)
        if not media_payload:
            return
        payload = {
            "chat_id": delivery["chat_id"],
            "media": media_payload,
            "business_connection_id": delivery.get("business_connection_id"),
        }
        self._call_api("sendMediaGroup", payload)

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


@dataclass
class NormalizedInboundEvent:
    event_type: str
    is_edited: bool
    business_connection_id: str | None
    business_account_id: int | None
    telegram_chat_id: int | None
    telegram_user_id: int | None
    telegram_message_id: int | None
    telegram_update_id: int | None
    text: str | None
    message_type: str | None
    callback_data: str | None
    raw_payload: dict[str, Any]
    occurred_at: datetime | None
    tg_business_account_id: str | None = None
    tg_user_id: str | None = None
    sender_type: str = "user"
    media_url: str | None = None
    idempotency_key: str | None = None


def _detect_telegram_message_type(message_obj: dict[str, Any]) -> str:
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


def normalize_telegram_update(raw_update: dict[str, Any]) -> NormalizedInboundEvent | None:
    update_id = raw_update.get("update_id")
    callback_query = raw_update.get("callback_query") or {}
    if callback_query:
        message = callback_query.get("message") or {}
        chat = message.get("chat") or {}
        from_user = callback_query.get("from") or {}
        return NormalizedInboundEvent(
            event_type="admin_callback",
            is_edited=False,
            business_connection_id=_extract_business_connection_id(raw_update),
            business_account_id=None,
            telegram_chat_id=int(chat.get("id")) if chat.get("id") is not None else None,
            telegram_user_id=int(from_user.get("id")) if from_user.get("id") is not None else None,
            telegram_message_id=int(message.get("message_id")) if message.get("message_id") is not None else None,
            telegram_update_id=int(update_id) if update_id is not None else None,
            text=None,
            message_type="callback",
            callback_data=str(callback_query.get("data") or ""),
            raw_payload=raw_update,
            occurred_at=utc_now(),
        )

    if raw_update.get("business_message"):
        message_obj = raw_update["business_message"]
        event_type = "business_message"
        is_edited = False
    elif raw_update.get("edited_business_message"):
        message_obj = raw_update["edited_business_message"]
        event_type = "edited_business_message"
        is_edited = True
    else:
        message_obj = raw_update.get("message") or {}
        chat = message_obj.get("chat") or {}
        if not message_obj or raw_update.get("business_connection"):
            return None
        if chat.get("type") != "private":
            return None
        event_type = "admin_text"
        is_edited = False

    chat = message_obj.get("chat") or {}
    from_user = message_obj.get("from") or {}
    business_connection_id = _extract_business_connection_id(raw_update)
    text_value = message_obj.get("text") or message_obj.get("caption")
    sent_at = utc_now()
    if isinstance(message_obj.get("date"), (int, float)):
        sent_at = datetime.fromtimestamp(message_obj["date"], tz=timezone.utc)
    normalized_payload = dict(raw_update)
    normalized_payload["business_connection_id"] = business_connection_id
    normalized_payload["chat_id"] = chat.get("id")
    normalized_payload["message_id"] = message_obj.get("message_id")
    tg_user_id = str(from_user.get("id") or chat.get("id") or "0")
    return NormalizedInboundEvent(
        event_type=event_type,
        is_edited=is_edited,
        business_connection_id=business_connection_id,
        business_account_id=None,
        telegram_chat_id=int(chat.get("id")) if chat.get("id") is not None else None,
        telegram_user_id=int(from_user.get("id")) if from_user.get("id") is not None else None,
        telegram_message_id=int(message_obj.get("message_id")) if message_obj.get("message_id") is not None else None,
        telegram_update_id=int(update_id) if update_id is not None else None,
        text=text_value,
        message_type=_detect_telegram_message_type(message_obj),
        callback_data=None,
        raw_payload=normalized_payload,
        occurred_at=sent_at,
        tg_business_account_id=business_connection_id or "default_business_connection",
        tg_user_id=tg_user_id,
    )


def build_event_idempotency_key(event: NormalizedInboundEvent) -> str:
    return ":".join([
        event.business_connection_id or "na",
        event.event_type,
        str(event.telegram_chat_id or 0),
        str(event.telegram_message_id or 0),
        "1" if event.is_edited else "0",
        str(event.telegram_update_id or 0),
    ])


class TelegramBusinessSenderAdapter:
    def __init__(self, tg_client) -> None:
        self.tg_client = tg_client

    def send_text(self, conversation_id: int, text: str) -> None:
        self.tg_client.send_text(conversation_id=conversation_id, text=text)


class SenderService:
    def __init__(self, outbound_job_repo: OutboundJobRepository, immediate_sender: TelegramBusinessSenderAdapter | None = None) -> None:
        self.outbound_job_repo = outbound_job_repo
        self.immediate_sender = immediate_sender

    def send_text_reply(self, conversation_id: int, text: str, delay_seconds: int = 0,
                        business_account_id: int | None = None, target_chat_id: int | None = None,
                        target_user_id: int | None = None, source_inbound_message_id: int | None = None,
                        delivery_plan: dict[str, Any] | None = None) -> None:
        delay_seconds = max(0, int(delay_seconds or 0))
        plan = delivery_plan or {"plan_version": 1, "steps": [], "summary": ""}
        has_followup_steps = bool(plan.get("steps"))
        if self.immediate_sender is not None and not has_followup_steps and delay_seconds <= 3:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            self.immediate_sender.send_text(conversation_id, text)
            logger.info("AI reply delivered immediately | conversation_id=%s", conversation_id)
            return
        send_after = utc_now() + timedelta(seconds=delay_seconds)
        dedup_key = f"text_reply:{source_inbound_message_id}" if source_inbound_message_id else None
        self.outbound_job_repo.enqueue(
            job_type="text_reply",
            business_account_id=business_account_id or 0,
            target_chat_id=target_chat_id or 0,
            target_user_id=target_user_id,
            payload={
                "conversation_id": conversation_id,
                "text": text,
                "delivery_plan": plan,
            },
            send_after=send_after,
            dedup_key=dedup_key,
            source_inbound_message_id=source_inbound_message_id,
        )
        logger.info("AI reply enqueued | conversation_id=%s | send_after=%s", conversation_id, send_after.isoformat())


class OutboundSenderWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run_forever, name="outbound-sender-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _execute_delivery_plan(self, tg_client: TelegramBotAPIClient, conversation_id: int, delivery_plan: dict[str, Any] | None) -> None:
        plan = delivery_plan or {}
        for step in plan.get("steps") or []:
            step_type = str(step.get("step_type") or "").strip().lower()
            if step_type == "text":
                text_value = str(step.get("text") or "").strip()
                buttons = tg_client.build_inline_keyboard(step.get("buttons") or [])
                if text_value or buttons:
                    tg_client.send_text(conversation_id, text_value or "More details here.", reply_markup=buttons)
            elif step_type == "photo":
                buttons = tg_client.build_inline_keyboard(step.get("buttons") or [])
                tg_client.send_photo(conversation_id, str(step.get("media_url") or ""), caption=step.get("caption"), reply_markup=buttons)
            elif step_type == "video":
                buttons = tg_client.build_inline_keyboard(step.get("buttons") or [])
                tg_client.send_video(conversation_id, str(step.get("media_url") or ""), caption=step.get("caption"), reply_markup=buttons)
            elif step_type == "media_group":
                media_items = list(step.get("media") or [])
                if media_items:
                    tg_client.send_media_group(conversation_id, media_items)
            else:
                logger.warning("Unsupported delivery step_type=%s", step_type)
            time.sleep(0.35)

    def _process_followup_plans(self, followup_repo: ProactiveFollowupPlanRepository, pacing_repo: PacingStateRepository,
                                runtime_policy_repo: RuntimePolicyRepository, conversation_repo: ConversationRepository,
                                user_control_repo: UserControlRepository, tg_client: TelegramBotAPIClient) -> None:
        work_hours_engine = WorkHoursEngine(self.settings.default_timezone)
        for plan in followup_repo.fetch_due(10):
            plan_id = int(plan['id'])
            followup_repo.mark_processing(plan_id)
            try:
                conversation_id = int(plan.get('conversation_id') or 0)
                if not conversation_id:
                    raise RuntimeError('missing conversation_id in followup plan')
                context = conversation_repo.get_context(conversation_id)
                if context.manual_takeover_status in ('active', 'pending_resume'):
                    followup_repo.reschedule(plan_id, utc_now() + timedelta(hours=6), 'manual takeover state delays followup')
                    continue
                control = user_control_repo.get_user_ai_control(conversation_id)
                if control and (control.get('manual_takeover_forced') or is_silence_active(control.get('silence_until'))):
                    followup_repo.reschedule(plan_id, utc_now() + timedelta(hours=6), 'user control delays followup')
                    continue
                policy = runtime_policy_repo.get_policy(int(plan.get('business_account_id') or context.business_account_id))
                if not policy.get('proactive_followup_enabled', True):
                    followup_repo.mark_failed(plan_id, 'followup disabled by policy', retryable=False)
                    continue
                if str(policy.get('proactive_followup_policy') or 'workhours_only') == 'workhours_only' and not work_hours_engine.is_in_reply_window(policy):
                    followup_repo.reschedule(plan_id, work_hours_engine.next_open_time(policy), 'rescheduled to next work window')
                    continue
                payload = dict(plan.get('payload_json') or {})
                text_value = str(payload.get('text') or '').strip()
                if not text_value:
                    raise RuntimeError('empty followup text')
                tg_client.send_text(conversation_id=conversation_id, text=text_value)
                conversation_repo.save_message(conversation_id, 'ai', 'text', text_value, {
                    'proactive_followup_plan_id': plan_id,
                    'followup_kind': plan.get('followup_kind'),
                    'followup_payload': payload,
                }, None)
                conversation_repo.set_last_ai_reply_at(conversation_id)
                pacing_repo.upsert_state(conversation_id, str(payload.get('pacing_mode') or 'light_presence'), None, None, None, 'low', {
                    'last_followup_kind': plan.get('followup_kind'),
                    'last_followup_reason': plan.get('reason_text'),
                }, False, True)
                followup_repo.mark_sent(plan_id)
            except Exception as plan_exc:
                logger.exception('Proactive followup plan failed | plan_id=%s', plan_id)
                followup_repo.mark_failed(plan_id, str(plan_exc), retryable=True)

    def run_forever(self) -> None:
        worker_db = Database(self.settings.database_url)
        worker_db.connect()
        job_repo = OutboundJobRepository(worker_db)
        tg_client = TelegramBotAPIClient(bot_token=self.settings.tg_bot_token, db=worker_db, admin_chat_ids=self.settings.admin_chat_ids)
        followup_repo = ProactiveFollowupPlanRepository(worker_db)
        pacing_repo = PacingStateRepository(worker_db)
        runtime_policy_repo = RuntimePolicyRepository(worker_db, self.settings.default_timezone)
        conversation_repo = ConversationRepository(worker_db)
        user_control_repo = UserControlRepository(worker_db)
        while not self._stop_event.is_set():
            try:
                self._process_followup_plans(followup_repo, pacing_repo, runtime_policy_repo, conversation_repo, user_control_repo, tg_client)
                jobs = job_repo.fetch_due(20)
                if not jobs:
                    time.sleep(0.5)
                    continue
                for job in jobs:
                    job_id = int(job['id'])
                    job_repo.mark_sending(job_id)
                    try:
                        payload = job.get('payload_json') or {}
                        if job.get('job_type') == 'text_reply':
                            conversation_id = int(payload.get('conversation_id') or 0)
                            text_value = str(payload.get('text') or '').strip()
                            delivery_plan = payload.get('delivery_plan') or {'plan_version': 1, 'steps': [], 'summary': ''}
                            if conversation_id and text_value:
                                tg_client.send_text(conversation_id=conversation_id, text=text_value)
                                self._execute_delivery_plan(tg_client, conversation_id, delivery_plan)
                                pacing_repo.upsert_state(conversation_id, 'normal', None, None, None, 'low', {'last_send_type': 'text_reply'}, False, True)
                            else:
                                raise RuntimeError('invalid text reply payload')
                        elif job.get('job_type') == 'admin_receipt':
                            text_value = str(payload.get('text') or '').strip()
                            if not text_value:
                                raise RuntimeError('invalid admin receipt payload')
                            tg_client.send_admin_text(int(job.get('target_chat_id') or 0), text_value)
                        else:
                            raise RuntimeError(f"unsupported job type: {job.get('job_type')}")
                        job_repo.mark_sent(job_id)
                    except Exception as job_exc:
                        logger.exception('Outbound sender job failed | job_id=%s', job_id)
                        job_repo.mark_failed(job_id, str(job_exc), retryable=True)
            except Exception:
                logger.exception('Outbound sender worker loop error')
                time.sleep(1.0)


class AdminNotifier:
    def __init__(self, tg_client, admin_chat_ids: list[int] | None = None) -> None:
        self.tg_client = tg_client
        self.admin_chat_ids = admin_chat_ids or []

    def notify_high_intent(self, business_account_id: int, user_id: int, conversation_id: int | None, level: str, reason: str) -> None:
        if not self.admin_chat_ids:
            return
        text = (
            "🔥 高意向提醒\n\n"
            f"业务账号ID: {business_account_id}\n"
            f"用户ID: {user_id}\n"
            f"会话ID: {conversation_id or '-'}\n"
            f"级别: {level}\n"
            f"原因: {reason}"
        )
        for chat_id in self.admin_chat_ids:
            self.tg_client.send_admin_text(chat_id=chat_id, text=text)


# =========================
# gateway
# =========================

class TelegramUpdateMapper:
    def map_update(self, raw_update: dict) -> InboundMessage | None:
        event = normalize_telegram_update(raw_update)
        if not event or event.event_type != "business_message":
            return None
        return InboundMessage(
            business_account_id=event.business_account_id or 0,
            tg_business_account_id=event.tg_business_account_id or "default_business_connection",
            user_id=int(event.telegram_user_id or event.telegram_chat_id or 0),
            tg_user_id=str(event.tg_user_id or event.telegram_user_id or event.telegram_chat_id or "0"),
            conversation_id=None,
            sender_type="user",
            message_type=event.message_type or "text",
            text=event.text,
            media_url=event.media_url,
            raw_payload=event.raw_payload,
            sent_at=event.occurred_at or utc_now(),
        )


class TelegramBusinessGateway:
    def __init__(self, settings: Settings, message_handler, business_account_repo: BusinessAccountRepository,
                 event_lock_repo: WebhookEventLockRepository, edit_repo: BusinessMessageEditRepository) -> None:
        self.settings = settings
        self.message_handler = message_handler
        self.business_account_repo = business_account_repo
        self.event_lock_repo = event_lock_repo
        self.edit_repo = edit_repo

    def handle_raw_update(self, raw_update: dict) -> str:
        event = normalize_telegram_update(raw_update)
        if event is None:
            return "ignored"
        if event.event_type in ("admin_callback", "admin_text"):
            return event.event_type
        event.idempotency_key = build_event_idempotency_key(event)
        account = self.business_account_repo.resolve_from_connection(event.business_connection_id)
        event.business_account_id = None if not account else int(account["id"])
        locked = self.event_lock_repo.acquire(
            event.idempotency_key,
            event.event_type,
            event.business_account_id,
            event.business_connection_id,
            event.telegram_chat_id,
            event.telegram_user_id,
            event.telegram_message_id,
            event.telegram_update_id,
            event.is_edited,
        )
        if not locked:
            self.event_lock_repo.record_duplicate_skip(event.idempotency_key)
            return "duplicate"
        try:
            if event.event_type == "edited_business_message":
                self.edit_repo.record_edit(
                    business_account_id=event.business_account_id,
                    telegram_chat_id=int(event.telegram_chat_id or 0),
                    telegram_user_id=event.telegram_user_id,
                    telegram_message_id=int(event.telegram_message_id or 0),
                    edited_text=event.text,
                    message_type=event.message_type,
                    raw_payload=event.raw_payload,
                )
                self.event_lock_repo.mark_processed(event.idempotency_key)
                return "edited_business_message"

            inbound = InboundMessage(
                business_account_id=event.business_account_id or 0,
                tg_business_account_id=event.tg_business_account_id or "default_business_connection",
                user_id=int(event.telegram_user_id or event.telegram_chat_id or 0),
                tg_user_id=str(event.tg_user_id or event.telegram_user_id or event.telegram_chat_id or "0"),
                conversation_id=None,
                sender_type="user",
                message_type=event.message_type or "text",
                text=event.text,
                media_url=event.media_url,
                raw_payload=event.raw_payload,
                sent_at=event.occurred_at or utc_now(),
            )
            self.message_handler(inbound)
            self.event_lock_repo.mark_processed(event.idempotency_key)
            return "business_message"
        except Exception as exc:
            self.event_lock_repo.mark_failed(event.idempotency_key, str(exc))
            raise
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
        persona_materials = self.material_repo.get_persona_materials(business_account_id)
        daily_materials = self.material_repo.get_daily_materials(business_account_id)
        bio_short_parts: list[str] = []
        self_share_topics: list[str] = []
        daily_life_style: list[str] = []
        media_package_hints: list[str] = []

        for item in persona_materials:
            material_type = (item.get("material_type") or "").strip().lower()
            content_text = item.get("content_text") or ""
            scene_tags = item.get("scene_tags_json") or []
            if material_type in ("intro", "resume") and content_text:
                bio_short_parts.append(content_text[:200])
            if material_type == "daily":
                if content_text:
                    self_share_topics.append(content_text[:120])
                if scene_tags:
                    daily_life_style.extend(str(i) for i in scene_tags)

        for item in daily_materials:
            content_text = item.get("content_text") or ""
            scene_tags = item.get("scene_tags_json") or []
            media_url = item.get("media_url") or ""
            if content_text:
                self_share_topics.append(content_text[:120])
            if scene_tags:
                daily_life_style.extend(str(i) for i in scene_tags)
            if media_url:
                media_package_hints.append((item.get("category") or "daily")[:40])

        return {
            "public_name": display_name,
            "nickname": display_name,
            "bio_short": " | ".join(bio_short_parts[:2]).strip() or f"{display_name} is a warm, low-pressure financial planning advisor.",
            "bio_long": " ".join(bio_short_parts[:4]).strip(),
            "self_share_topics": self_share_topics[:10],
            "daily_life_style": daily_life_style[:14],
            "daily_media_hints": media_package_hints[:6],
        }

    @staticmethod
    def to_summary(profile: dict) -> str:
        return (
            f"Public name: {profile.get('public_name')}; Nickname: {profile.get('nickname')}; "
            f"Bio short: {profile.get('bio_short')}; Self share topics: {profile.get('self_share_topics')}; "
            f"Daily life style: {profile.get('daily_life_style')}; Daily media hints: {profile.get('daily_media_hints')}."
        )


class UserUnderstandingEngine:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def analyze(self, latest_user_message: str, recent_context: list[dict], persona_core_summary: str,
                user_state_summary: str, long_term_memory_summary: str = "") -> UnderstandingResult:
        fallback = self._fallback_rule_based(latest_user_message)
        try:
            result = self.llm_service.classify_user_message(
                build_understanding_prompt(latest_user_message, recent_context, persona_core_summary, user_state_summary, long_term_memory_summary)
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

    def generate(self, latest_user_message: str, recent_context: list[dict], persona_summary: str, user_state_summary: str,
                 stage: str, chat_mode: str, understanding: dict, reply_plan: dict, selected_content: dict,
                 memory_summary: dict[str, Any] | None = None, reply_language: str = "English") -> str:
        try:
            text = self.llm_service.generate_reply(build_reply_prompt(
                latest_user_message, recent_context, persona_summary, user_state_summary,
                stage, chat_mode, understanding, reply_plan, selected_content, memory_summary, reply_language,
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


class WorkHoursEngine:
    def __init__(self, default_timezone: str = 'UTC') -> None:
        self.default_timezone = default_timezone or 'UTC'

    def _timezone(self, policy: dict[str, Any] | None) -> str:
        return str((policy or {}).get('timezone_name') or self.default_timezone)

    def _weekdays(self, policy: dict[str, Any] | None) -> list[int]:
        weekdays = (policy or {}).get('active_weekdays_json') or [1, 2, 3, 4, 5, 6, 7]
        try:
            return [int(i) for i in weekdays]
        except Exception:
            return [1, 2, 3, 4, 5, 6, 7]

    def is_in_reply_window(self, policy: dict[str, Any] | None, at: datetime | None = None) -> bool:
        policy = policy or {}
        current_utc = at or utc_now()
        local_now = current_utc.astimezone(ZoneInfo(self._timezone(policy)))
        if int(local_now.isoweekday()) not in self._weekdays(policy):
            return False
        start_hour = int(policy.get('reply_window_start_hour') or 9)
        end_hour = int(policy.get('reply_window_end_hour') or 21)
        if start_hour == end_hour:
            return True
        current_minutes = local_now.hour * 60 + local_now.minute
        start_minutes = start_hour * 60
        end_minutes = end_hour * 60
        if start_minutes < end_minutes:
            return start_minutes <= current_minutes < end_minutes
        return current_minutes >= start_minutes or current_minutes < end_minutes

    def next_open_time(self, policy: dict[str, Any] | None, after: datetime | None = None) -> datetime:
        policy = policy or {}
        current_utc = after or utc_now()
        local_after = current_utc.astimezone(ZoneInfo(self._timezone(policy)))
        weekdays = self._weekdays(policy)
        start_hour = int(policy.get('reply_window_start_hour') or 9)
        for day_offset in range(0, 14):
            candidate_day = (local_after + timedelta(days=day_offset)).replace(minute=0, second=0, microsecond=0)
            if int(candidate_day.isoweekday()) not in weekdays:
                continue
            start_local = candidate_day.replace(hour=start_hour)
            if day_offset == 0:
                if self.is_in_reply_window(policy, current_utc):
                    return current_utc
                if local_after < start_local:
                    return start_local.astimezone(timezone.utc)
            else:
                return start_local.astimezone(timezone.utc)
        return (local_after + timedelta(hours=12)).astimezone(timezone.utc)

    def describe_window(self, policy: dict[str, Any] | None) -> str:
        policy = policy or {}
        tz_name = self._timezone(policy)
        start_hour = int(policy.get('reply_window_start_hour') or 9)
        end_hour = int(policy.get('reply_window_end_hour') or 21)
        return f"{start_hour:02d}:00-{end_hour:02d}:00 ({tz_name})"


class ProactiveFollowupEngine:
    def decide(self, policy: dict[str, Any], understanding: UnderstandingResult, mode_decision: ModeDecision,
               reply_plan: ReplyPlan, user_state: UserStateSnapshot, intent_decision: IntentDecision,
               context: ConversationContext, latest_user_text: str, project_name: str | None = None,
               latest_handover_summary: dict[str, Any] | None = None,
               escalation_decision: dict[str, Any] | None = None) -> FollowupDecision:
        if not (policy or {}).get('proactive_followup_enabled', True):
            return FollowupDecision(False, reason='followup disabled by policy', pacing_mode='disabled')
        if context.manual_takeover_status in ('active', 'pending_resume'):
            return FollowupDecision(False, reason='manual takeover state', pacing_mode='handover')
        if user_state.ops_category in ('archived_user', 'invalid_user', 'blacklist_user'):
            return FollowupDecision(False, reason=f'ops category {user_state.ops_category}', pacing_mode='silent')
        if escalation_decision and escalation_decision.get('notify_level') == 'urgent_takeover':
            return FollowupDecision(False, reason='urgent human takeover preferred', pacing_mode='human_first')

        now = utc_now()
        min_minutes = max(60, int((policy or {}).get('min_followup_delay_minutes') or 720))
        max_minutes = max(min_minutes, int((policy or {}).get('max_followup_delay_minutes') or 2160))
        busy_snooze_hours = max(4, int((policy or {}).get('busy_snooze_hours') or 18))
        stop_snooze_hours = max(busy_snooze_hours, int((policy or {}).get('stop_snooze_hours') or 72))
        normalized_project = str(project_name or '').strip()
        memory_hint = ''
        if latest_handover_summary:
            memory_hint = str(latest_handover_summary.get('resume_suggestion') or '').strip()

        if understanding.boundary_signal == 'stop':
            return FollowupDecision(False, 'do_not_push', None, 'hard stop boundary detected', '', 'quiet', now + timedelta(hours=stop_snooze_hours))

        if understanding.boundary_signal in ('busy', 'later', 'sleep'):
            scheduled_for = now + timedelta(hours=busy_snooze_hours)
            if understanding.boundary_signal == 'sleep':
                scheduled_for = max(scheduled_for, now + timedelta(hours=10))
            elif understanding.boundary_signal == 'later':
                scheduled_for = max(scheduled_for, now + timedelta(hours=14))
            text = "Hope your day is going smoothly. No rush at all. When you're free later, I can pick this up from where we paused."
            if normalized_project:
                text = f"No rush at all. When you're free later, I can continue the {normalized_project} part from where we paused."
            return FollowupDecision(True, 'boundary_resume', scheduled_for, f'boundary={understanding.boundary_signal}', text, 'respectful_pause', scheduled_for)

        if understanding.high_intent_signal or intent_decision.level in ('high', 'very_high'):
            scheduled_for = now + timedelta(minutes=min_minutes)
            text = 'Just checking in gently. When you have a moment, I can help you with the next part and keep it simple.'
            if normalized_project:
                text = f"Just checking in gently. When you have a moment, I can help you with the next {normalized_project} step and keep it simple."
            return FollowupDecision(True, 'high_intent_resume', scheduled_for, 'high intent followup', text, 'warm_active', None)

        if mode_decision.chat_mode == 'emotional':
            scheduled_for = now + timedelta(minutes=min(max_minutes, max(min_minutes, 18 * 60)))
            return FollowupDecision(True, 'emotional_checkin', scheduled_for, 'emotional support continuation', "Just a gentle check-in from my side. No pressure at all — I hope you're feeling a little lighter today.", 'soft_support', None)

        if user_state.ops_category == 'followup_user' or 'followup_worthy' in (user_state.tags or []):
            scheduled_for = now + timedelta(minutes=min(max_minutes, max(min_minutes, 24 * 60)))
            text = 'Just a light check-in from my side. If you want, we can continue whenever it feels natural for you.'
            if memory_hint:
                text = 'Just a light check-in from my side. No pressure — we can continue naturally whenever it suits you.'
            return FollowupDecision(True, 'gentle_checkin', scheduled_for, 'followup user cadence', text, 'light_presence', None)

        if reply_plan.goal in ('maintain', 'keep_replying') and latest_user_text.strip():
            scheduled_for = now + timedelta(minutes=min(max_minutes, max(min_minutes, 30 * 60)))
            return FollowupDecision(True, 'daily_touch', scheduled_for, 'relationship maintenance cadence', 'Just saying hi from my side. How has your day been going?', 'light_presence', None)

        return FollowupDecision(False, reason='no proactive followup needed', pacing_mode='normal')


class AISwitchEngine:
    def __init__(self, settings_repo: SettingsRepository, user_control_repo: UserControlRepository,
                 runtime_policy_repo: RuntimePolicyRepository | None = None,
                 work_hours_engine: WorkHoursEngine | None = None) -> None:
        self.settings_repo = settings_repo
        self.user_control_repo = user_control_repo
        self.runtime_policy_repo = runtime_policy_repo
        self.work_hours_engine = work_hours_engine or WorkHoursEngine()

    def decide(self, business_account_id: int, conversation_id: int, ops_category: str, manual_takeover_status: str) -> tuple[bool, str]:
        if not self.settings_repo.get_global_ai_enabled(business_account_id):
            return False, "global ai disabled"
        if not self.settings_repo.get_ops_category_ai_enabled(business_account_id, ops_category):
            return False, f"ops category disabled: {ops_category}"
        policy = self.runtime_policy_repo.get_policy(business_account_id) if self.runtime_policy_repo else {}
        if str(policy.get('inbound_ai_policy') or 'always') == 'workhours_only' and not self.work_hours_engine.is_in_reply_window(policy):
            return False, 'outside inbound work hours'
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
        "daily": ["daily_warmup", "small_talk"],
        "maintain": ["maintain", "followup"],
        "product": ["product_intro", "faq", "proof", "process", "returns"],
        "high_intent": ["high_intent", "process", "faq", "proof"],
        "emotional": ["comfort", "maintain"],
        "pause": [],
    }

    def __init__(self, script_repo: ScriptRepository) -> None:
        self.script_repo = script_repo

    def select(self, project_id: int | None, mode: str) -> list[dict]:
        if not project_id or mode not in self.CATEGORY_MAP:
            return []
        return self.script_repo.get_project_scripts(project_id, self.CATEGORY_MAP[mode])[:3]


class MaterialSelector:
    MODE_KEYWORDS = {
        "daily": {"daily", "lifestyle", "intro", "warmup"},
        "maintain": {"maintain", "followup", "trust", "proof"},
        "emotional": {"emotional", "care", "support", "daily"},
        "product": {"product", "faq", "proof", "yield", "returns", "process"},
        "high_intent": {"product", "faq", "proof", "yield", "returns", "process", "join", "start"},
        "pause": set(),
    }

    def __init__(self, material_repo: MaterialRepository) -> None:
        self.material_repo = material_repo

    def _score_item(self, item: dict[str, Any], mode: str, should_send_material: bool) -> float:
        if not should_send_material and mode not in ("daily", "maintain", "emotional"):
            return -999
        tags = {str(i).strip().lower() for i in (item.get("scene_tags_json") or []) if str(i).strip()}
        material_type = str(item.get("material_type") or item.get("category") or "").strip().lower()
        score = 0.0
        if mode in ("product", "high_intent"):
            score += 3.0
        if material_type in self.MODE_KEYWORDS.get(mode, set()):
            score += 3.0
        if tags & self.MODE_KEYWORDS.get(mode, set()):
            score += 2.0
        if item.get("media_url"):
            score += 0.5
        priority = int(item.get("priority") or 100)
        score += max(0.0, (120 - min(priority, 120)) / 120.0)
        return score

    def select(self, items: list[dict[str, Any]], mode: str, should_send_material: bool, limit: int = 3) -> list[dict]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            score = self._score_item(item, mode, should_send_material)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: (-x[0], int(x[1].get("priority") or 100), int(x[1].get("id") or 0)))
        return [item for _, item in scored[:limit]]


class PersonaMaterialSelector:
    def select(self, persona_materials: list[dict], mode: str, limit: int = 3) -> list[dict]:
        desired = {"daily"} if mode in ("daily", "maintain", "emotional") else set()
        if not desired:
            return []
        picked: list[dict] = []
        for item in persona_materials:
            material_type = str(item.get("material_type") or "").lower()
            if material_type in desired or (mode == "daily" and material_type in ("intro", "resume")):
                picked.append(item)
            if len(picked) >= limit:
                break
        return picked


class DailyMaterialSelector:
    def select(self, daily_materials: list[dict], mode: str, limit: int = 3) -> list[dict]:
        if mode not in ("daily", "maintain", "emotional"):
            return []
        ranked: list[dict] = []
        keywords = {mode, "daily", "life", "story", "warm"}
        for item in daily_materials:
            tags = {str(i).strip().lower() for i in (item.get("scene_tags_json") or []) if str(i).strip()}
            category = str(item.get("category") or "").lower()
            score = 0
            if category in keywords:
                score += 2
            if tags & keywords:
                score += 2
            if item.get("media_url"):
                score += 1
            ranked.append({**item, "_score": score})
        ranked.sort(key=lambda x: (-int(x.get("_score") or 0), int(x.get("priority") or 100), int(x.get("id") or 0)))
        return [{k: v for k, v in item.items() if k != "_score"} for item in ranked[:limit]]


class MaterialPackageSelector:
    MODE_KEYWORDS = {
        "daily": {"daily", "lifestyle", "warmup", "story", "persona"},
        "maintain": {"maintain", "followup", "trust", "daily", "checkin"},
        "emotional": {"emotional", "care", "support", "daily"},
        "product": {"product", "faq", "proof", "yield", "returns", "process"},
        "high_intent": {"product", "faq", "proof", "yield", "returns", "process", "join", "start"},
        "pause": set(),
    }

    def __init__(self, material_repo: MaterialRepository) -> None:
        self.material_repo = material_repo

    def _score_package(self, package_row: dict[str, Any], mode: str, should_send_material: bool) -> float:
        scope = str(package_row.get("package_scope") or "").lower()
        kind = str(package_row.get("package_kind") or "").lower()
        tags = {str(i).strip().lower() for i in (package_row.get("scene_tags_json") or []) if str(i).strip()}
        keywords = self.MODE_KEYWORDS.get(mode, set())
        score = 0.0
        if scope == "project" and mode in ("product", "high_intent") and should_send_material:
            score += 5.0
        if scope in ("persona", "daily") and mode in ("daily", "maintain", "emotional"):
            score += 4.0
        if kind and kind in keywords:
            score += 3.0
        if tags & keywords:
            score += 2.0
        if package_row.get("description"):
            score += 0.25
        priority = int(package_row.get("priority") or 100)
        score += max(0.0, (120 - min(priority, 120)) / 120.0)
        return score

    def select(self, business_account_id: int, package_scope: str, project_id: int | None, mode: str, should_send_material: bool, limit: int = 1) -> list[dict]:
        if package_scope == "project" and not project_id:
            return []
        rows = self.material_repo.list_material_packages(business_account_id, package_scope, project_id)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            score = self._score_package(row, mode, should_send_material)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: (-x[0], int(x[1].get("priority") or 100), int(x[1].get("id") or 0)))
        result: list[dict] = []
        for _, row in scored[:limit]:
            result.append(self.material_repo.build_material_package(row))
        return result


def _summarize_material_items(items: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items[:limit]:
        compact.append({
            "id": item.get("id"),
            "type": item.get("item_type") or item.get("material_type") or item.get("category"),
            "text": (item.get("content_text") or item.get("caption_text") or "")[:120],
            "has_media": bool(item.get("media_url")),
            "media_url": item.get("media_url"),
        })
    return compact


def _summarize_package_for_prompt(package_row: dict[str, Any]) -> dict[str, Any]:
    items = package_row.get("items") or []
    buttons = package_row.get("buttons") or []
    return {
        "id": package_row.get("id"),
        "scope": package_row.get("package_scope"),
        "name": package_row.get("package_name"),
        "kind": package_row.get("package_kind"),
        "description": (package_row.get("description") or "")[:160],
        "allow_partial": bool(package_row.get("allow_partial")),
        "items": _summarize_material_items(items, 5),
        "buttons": [
            {
                "text": button.get("button_text"),
                "type": button.get("button_type"),
                "value": button.get("button_value"),
            }
            for button in buttons[:4]
        ],
    }


class ContentSelector:
    def __init__(self, material_repo: MaterialRepository, script_selector: ScriptSelector, material_selector: MaterialSelector, persona_material_selector: PersonaMaterialSelector, daily_material_selector: DailyMaterialSelector, material_package_selector: MaterialPackageSelector) -> None:
        self.material_repo = material_repo
        self.script_selector = script_selector
        self.material_selector = material_selector
        self.persona_material_selector = persona_material_selector
        self.daily_material_selector = daily_material_selector
        self.material_package_selector = material_package_selector

    def _build_prompt_summary(self, selected: dict[str, Any]) -> str:
        lines: list[str] = []
        if selected.get("project_scripts"):
            lines.append(f"Project scripts: {len(selected['project_scripts'])}")
        if selected.get("project_materials"):
            lines.append(f"Project materials: {len(selected['project_materials'])}")
        if selected.get("persona_materials"):
            lines.append(f"Persona materials: {len(selected['persona_materials'])}")
        if selected.get("daily_materials"):
            lines.append(f"Daily materials: {len(selected['daily_materials'])}")
        package_groups = selected.get("material_packages") or {}
        for scope_key in ("project", "persona", "daily"):
            packages = package_groups.get(scope_key) or []
            for package in packages:
                lines.append(f"{scope_key} package: {package.get('package_name')} | items={len(package.get('items') or [])} | buttons={len(package.get('buttons') or [])}")
        return " ; ".join(lines)[:600]

    def select(self, business_account_id: int, project_id: int | None, mode: str, reply_plan: ReplyPlan) -> dict:
        persona_materials = self.material_repo.get_persona_materials(business_account_id)
        daily_materials = self.material_repo.get_daily_materials(business_account_id)
        project_materials = self.material_repo.get_project_materials(project_id) if project_id else []
        selected = {
            "persona_materials": self.persona_material_selector.select(persona_materials, mode),
            "daily_materials": self.daily_material_selector.select(daily_materials, mode),
            "project_scripts": self.script_selector.select(project_id, mode),
            "project_materials": self.material_selector.select(project_materials, mode, reply_plan.should_send_material),
            "material_packages": {
                "project": self.material_package_selector.select(business_account_id, "project", project_id, mode, reply_plan.should_send_material, limit=1),
                "persona": self.material_package_selector.select(business_account_id, "persona", project_id, mode, reply_plan.should_send_material, limit=1),
                "daily": self.material_package_selector.select(business_account_id, "daily", project_id, mode, reply_plan.should_send_material, limit=1),
            },
        }
        selected["material_library_overview"] = self.material_repo.get_material_library_overview(business_account_id, project_id)
        selected["prompt_packages"] = {
            scope: [_summarize_package_for_prompt(pkg) for pkg in packages]
            for scope, packages in (selected.get("material_packages") or {}).items()
        }
        selected["prompt_summary"] = self._build_prompt_summary(selected)
        return selected


def _infer_delivery_media_type(item_type: str | None, media_url: str | None) -> str:
    normalized = str(item_type or "").strip().lower()
    if normalized in {"video", "mp4", "clip", "movie"}:
        return "video"
    url = str(media_url or "").lower()
    for suffix in (".mp4", ".mov", ".m4v", ".webm", ".mkv"):
        if suffix in url:
            return "video"
    return "photo"


def _normalize_delivery_buttons(buttons: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for button in buttons or []:
        text_value = str(button.get("button_text") or button.get("text") or "").strip()
        value = str(button.get("button_value") or button.get("value") or "").strip()
        button_type = str(button.get("button_type") or button.get("type") or "url").strip().lower()
        if not text_value or not value:
            continue
        normalized.append({"text": text_value, "value": value, "type": button_type})
    return normalized


def _make_text_delivery_step(text_value: str, origin: dict[str, Any], buttons: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    normalized_text = str(text_value or "").strip()
    normalized_buttons = _normalize_delivery_buttons(buttons)
    if not normalized_text and not normalized_buttons:
        return None
    return {
        "step_type": "text",
        "text": normalized_text,
        "buttons": normalized_buttons,
        "origin": origin,
    }


def _make_media_delivery_step(media_url: str, media_type: str, caption_text: str | None, origin: dict[str, Any]) -> dict[str, Any] | None:
    normalized_media = str(media_url or "").strip()
    if not normalized_media:
        return None
    return {
        "step_type": media_type,
        "media_url": normalized_media,
        "caption": str(caption_text or "").strip() or None,
        "origin": origin,
    }


def _delivery_steps_from_material_item(item: dict[str, Any], scope: str, package_row: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    origin = {
        "scope": scope,
        "material_id": item.get("id"),
        "package_id": None if package_row is None else package_row.get("id"),
        "package_name": None if package_row is None else package_row.get("package_name"),
    }
    item_type = str(item.get("item_type") or item.get("material_type") or item.get("category") or "").strip().lower()
    text_value = str(item.get("content_text") or item.get("caption_text") or "").strip()
    media_url = str(item.get("media_url") or "").strip()
    steps: list[dict[str, Any]] = []
    if media_url:
        media_type = _infer_delivery_media_type(item_type, media_url)
        media_step = _make_media_delivery_step(media_url, media_type, item.get("caption_text") or item.get("content_text"), origin)
        if media_step:
            steps.append(media_step)
        extra_text = str(item.get("content_text") or "").strip()
        if item.get("caption_text"):
            extra_text = ""
        if extra_text and len(extra_text) > 220:
            text_step = _make_text_delivery_step(extra_text, origin)
            if text_step:
                steps.append(text_step)
    else:
        text_step = _make_text_delivery_step(text_value, origin)
        if text_step:
            steps.append(text_step)
    return steps


def _coalesce_delivery_steps(raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_steps: list[dict[str, Any]] = []
    media_buffer: list[dict[str, Any]] = []

    def flush_media_buffer() -> None:
        nonlocal media_buffer
        if not media_buffer:
            return
        if len(media_buffer) == 1:
            final_steps.append(media_buffer[0])
        else:
            media_items: list[dict[str, Any]] = []
            for idx, step in enumerate(media_buffer[:10]):
                media_items.append({
                    "type": "video" if step.get("step_type") == "video" else "photo",
                    "media_url": step.get("media_url"),
                    "caption": step.get("caption") if idx == 0 else None,
                })
            final_steps.append({
                "step_type": "media_group",
                "media": media_items,
                "origin": {"grouped_from": [step.get("origin") for step in media_buffer[:10]]},
            })
        media_buffer = []

    for step in raw_steps:
        if step.get("step_type") in {"photo", "video"}:
            media_buffer.append(step)
            if len(media_buffer) >= 8:
                flush_media_buffer()
            continue
        flush_media_buffer()
        final_steps.append(step)
    flush_media_buffer()
    return final_steps


def _build_package_delivery_steps(package_row: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for item in package_row.get("items") or []:
        steps.extend(_delivery_steps_from_material_item(item, scope, package_row=package_row))
    button_step = _make_text_delivery_step(
        str(package_row.get("description") or package_row.get("package_name") or "More details here."),
        {"scope": scope, "package_id": package_row.get("id"), "package_name": package_row.get("package_name")},
        buttons=package_row.get("buttons") or [],
    )
    if button_step and button_step.get("buttons"):
        steps.append(button_step)
    return steps


def _summarize_delivery_plan(delivery_plan: dict[str, Any] | None) -> str:
    plan = delivery_plan or {}
    steps = plan.get("steps") or []
    parts: list[str] = []
    for step in steps[:8]:
        step_type = str(step.get("step_type") or "unknown")
        if step_type == "media_group":
            parts.append(f"media_group:{len(step.get('media') or [])}")
        elif step_type == "text":
            parts.append("text")
        else:
            parts.append(step_type)
    return ", ".join(parts)


def build_material_delivery_plan(selected_content: dict[str, Any], should_send_material: bool) -> dict[str, Any]:
    if not should_send_material:
        return {"plan_version": 1, "steps": [], "summary": ""}

    package_groups = selected_content.get("material_packages") or {}
    raw_steps: list[dict[str, Any]] = []
    used_scope_packages: set[str] = set()

    for scope in ("project", "persona", "daily"):
        packages = package_groups.get(scope) or []
        if packages:
            used_scope_packages.add(scope)
            raw_steps.extend(_build_package_delivery_steps(packages[0], scope))

    scope_to_material_key = {
        "project": "project_materials",
        "persona": "persona_materials",
        "daily": "daily_materials",
    }
    for scope, material_key in scope_to_material_key.items():
        if scope in used_scope_packages:
            continue
        for item in (selected_content.get(material_key) or [])[:2]:
            raw_steps.extend(_delivery_steps_from_material_item(item, scope))

    final_steps = _coalesce_delivery_steps(raw_steps)[:8]
    delivery_plan = {
        "plan_version": 1,
        "steps": final_steps,
        "summary": _summarize_delivery_plan({"steps": final_steps}),
    }
    return delivery_plan



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



class ConversationMemoryManager:
    def __init__(self, conversation_repo: ConversationRepository, llm_service: LLMService) -> None:
        self.conversation_repo = conversation_repo
        self.llm_service = llm_service

    @staticmethod
    def build_memory_summary_text(long_term_memory: dict[str, Any] | None, latest_handover_summary: dict[str, Any] | None = None) -> str:
        memory = long_term_memory or {}
        if not memory and not latest_handover_summary:
            return ""
        parts: list[str] = []
        summary_text = str(memory.get("summary_text") or "").strip()
        if summary_text:
            parts.append(f"RollingSummary={summary_text}")
        key_facts = memory.get("key_facts") or []
        if key_facts:
            parts.append(f"KeyFacts={'; '.join(str(i) for i in key_facts[:5])}")
        preferences = memory.get("user_preferences") or []
        if preferences:
            parts.append(f"Preferences={'; '.join(str(i) for i in preferences[:4])}")
        boundaries = memory.get("boundaries") or []
        if boundaries:
            parts.append(f"Boundaries={'; '.join(str(i) for i in boundaries[:4])}")
        project_signals = memory.get("project_signals") or []
        if project_signals:
            parts.append(f"ProjectSignals={'; '.join(str(i) for i in project_signals[:4])}")
        followup_strategy = str(memory.get("followup_strategy") or "").strip()
        if followup_strategy:
            parts.append(f"Followup={followup_strategy}")
        if latest_handover_summary:
            resume_suggestion = str(latest_handover_summary.get("resume_suggestion") or "").strip()
            human_strategy = str(latest_handover_summary.get("human_strategy_summary") or "").strip()
            if resume_suggestion:
                parts.append(f"ResumeHint={resume_suggestion}")
            if human_strategy:
                parts.append(f"HumanStrategy={human_strategy}")
        return " | ".join(parts)

    def should_refresh_summary(self, context: ConversationContext, understanding: UnderstandingResult, tag_result: dict[str, Any] | None = None) -> bool:
        if not context.recent_summary:
            return True
        if len(context.recent_messages or []) >= 10:
            return True
        if understanding.high_intent_signal or understanding.explicit_product_query or bool(understanding.boundary_signal):
            return True
        if tag_result and ((tag_result.get("added") or []) or (tag_result.get("removed") or [])):
            return True
        return False

    def build_fallback_summary(self, latest_user_message: str, understanding: UnderstandingResult, context: ConversationContext,
                               state_snapshot: dict[str, Any], latest_reply: str | None = None) -> dict[str, Any]:
        project_name = str(state_snapshot.get("project_name") or "").strip()
        segment_name = str(state_snapshot.get("segment_name") or "").strip()
        tags = state_snapshot.get("tags") or []
        boundaries: list[str] = []
        if understanding.boundary_signal:
            boundaries.append(str(understanding.boundary_signal))
        if understanding.resistance_signal:
            boundaries.append(f"resistance:{understanding.resistance_signal}")
        project_signals: list[str] = []
        if project_name:
            project_signals.append(f"project={project_name}")
        if segment_name:
            project_signals.append(f"segment={segment_name}")
        if understanding.need_type and understanding.need_type != "conversation":
            project_signals.append(f"need={understanding.need_type}")
        summary_text = (
            f"User is currently in stage {state_snapshot.get('stage') or context.current_stage or '-'} "
            f"with mode {state_snapshot.get('chat_mode') or context.current_chat_mode or '-'}; "
            f"latest need={understanding.need_type}; latest message={latest_user_message[:120]}"
        ).strip()
        return {
            "summary_text": summary_text,
            "key_facts": [f"ops={state_snapshot.get('ops_category') or '-'}", f"intent={state_snapshot.get('intent_level') or '-'}"],
            "user_preferences": [],
            "boundaries": boundaries,
            "project_signals": project_signals,
            "followup_strategy": str(state_snapshot.get("reply_goal") or "keep a natural next turn"),
            "relationship_notes": f"Keep pace natural. Tags: {', '.join(tags) if tags else '-'}",
            "tags_to_watch": tags[:6],
            "latest_reply": (latest_reply or "")[:200],
        }

    def refresh_summary(self, conversation_id: int, context: ConversationContext, latest_user_message: str,
                        understanding: UnderstandingResult, state_snapshot: dict[str, Any],
                        latest_reply: str | None = None, force: bool = False) -> dict[str, Any] | None:
        if not force and not self.should_refresh_summary(context, understanding, {"added": state_snapshot.get("tag_changes_added") or [], "removed": state_snapshot.get("tag_changes_removed") or []}):
            return None
        recent_messages = self.conversation_repo.get_messages_for_summary(conversation_id, 30)
        conversation_context = {
            "conversation_id": context.conversation_id,
            "business_account_id": context.business_account_id,
            "user_id": context.user_id,
            "current_stage": context.current_stage,
            "current_chat_mode": context.current_chat_mode,
            "current_mainline": context.current_mainline,
            "recent_summary": context.recent_summary,
            "existing_memory": context.long_term_memory or {},
        }
        try:
            summary_payload = self.llm_service.summarize_conversation(
                build_conversation_summary_prompt(
                    conversation_context,
                    recent_messages,
                    latest_user_message,
                    understanding.__dict__,
                    state_snapshot,
                    latest_reply=latest_reply,
                )
            )
        except Exception:
            summary_payload = self.build_fallback_summary(latest_user_message, understanding, context, state_snapshot, latest_reply)
        summary_text = str(summary_payload.get("summary_text") or "").strip()
        if not summary_text:
            summary_payload = self.build_fallback_summary(latest_user_message, understanding, context, state_snapshot, latest_reply)
            summary_text = str(summary_payload.get("summary_text") or "").strip()
        self.conversation_repo.save_conversation_summary(
            conversation_id=conversation_id,
            summary_text=summary_text,
            summary_json=summary_payload,
            summary_kind="rolling",
            source="ai",
            message_count=len(recent_messages),
        )
        return summary_payload




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
                 sender_service: SenderService, memory_manager: ConversationMemoryManager,
                 runtime_policy_repo: RuntimePolicyRepository,
                 pacing_state_repo: PacingStateRepository,
                 proactive_followup_repo: ProactiveFollowupPlanRepository,
                 work_hours_engine: WorkHoursEngine,
                 proactive_followup_engine: ProactiveFollowupEngine,
                 handover_repo: HandoverRepository | None = None,
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
        self.memory_manager = memory_manager
        self.runtime_policy_repo = runtime_policy_repo
        self.pacing_state_repo = pacing_state_repo
        self.proactive_followup_repo = proactive_followup_repo
        self.work_hours_engine = work_hours_engine
        self.proactive_followup_engine = proactive_followup_engine
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
        self.runtime_policy_repo.ensure_default_policy(business_account_id)
        user_payload = inbound_message.raw_payload.get("business_message") or inbound_message.raw_payload.get("edited_business_message") or inbound_message.raw_payload.get("message") or {}
        from_user = user_payload.get("from") or {}
        user_id = self.user_repo.get_or_create_user(
            inbound_message.tg_user_id,
            display_name=(from_user.get("first_name") or f"User-{inbound_message.tg_user_id}"),
            username=from_user.get("username"),
            language_code=from_user.get("language_code"),
        )
        conversation_id = self.conversation_repo.get_or_create_conversation(business_account_id, user_id)
        inbound_message_id = self.conversation_repo.save_message(
            conversation_id, "user", inbound_message.message_type, inbound_message.text, inbound_message.raw_payload, inbound_message.media_url
        )
        context = self.conversation_repo.get_context(conversation_id)
        user_profile = self.user_repo.get_user_profile(business_account_id, user_id)
        user_state = self.user_repo.get_user_state_snapshot(business_account_id, user_id)
        ai_allowed, ai_reason = self.ai_switch_engine.decide(business_account_id, conversation_id, user_state.ops_category, context.manual_takeover_status)
        if not ai_allowed:
            logger.info("AI reply skipped | conversation_id=%s | reason=%s", conversation_id, ai_reason)
            return

        latest_user_text = inbound_message.text or ""
        runtime_policy = self.runtime_policy_repo.get_policy(business_account_id)
        self.proactive_followup_repo.cancel_open_plans(conversation_id, 'new inbound message')
        existing_pacing_state = self.pacing_state_repo.get_state(conversation_id) or {}
        self.pacing_state_repo.upsert_state(conversation_id, str(existing_pacing_state.get('pacing_mode') or 'normal'), None, None, existing_pacing_state.get('followup_snooze_until'), 'low', {'last_inbound_text': latest_user_text[:120]}, True, False)
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

        memory_summary_text = self.memory_manager.build_memory_summary_text(context.long_term_memory, latest_handover_summary)
        understanding = self.understanding_engine.analyze(
            latest_user_text,
            recent_context,
            self.persona_core.to_summary(),
            self._user_state_summary(user_state),
            memory_summary_text,
        )
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
        if context.long_term_memory:
            understanding_payload["memory_summary"] = context.long_term_memory.get("summary_text")
            understanding_payload["known_boundaries"] = context.long_term_memory.get("boundaries")
            understanding_payload["known_preferences"] = context.long_term_memory.get("user_preferences")

        reply_language = resolve_user_reply_language(user_profile)
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
            memory_summary=context.long_term_memory,
            reply_language=reply_language,
        )
        final_text = self.reply_self_check_engine.check_and_fix(draft_reply, mode_decision.chat_mode, understanding)
        delay_seconds = self.reply_delay_engine.decide_delay_seconds(understanding, mode_decision.chat_mode, final_text)
        selected_package_ids: list[int] = []
        selected_material_ids: list[int] = []
        for item in (selected_content.get("project_materials") or []) + (selected_content.get("persona_materials") or []) + (selected_content.get("daily_materials") or []):
            if item.get("id") is not None:
                selected_material_ids.append(int(item.get("id")))
        for package_scope_items in (selected_content.get("material_packages") or {}).values():
            for package in package_scope_items or []:
                if package.get("id") is not None:
                    selected_package_ids.append(int(package.get("id")))
                for item in (package.get("items") or []):
                    if item.get("id") is not None:
                        selected_material_ids.append(int(item.get("id")))
        delivery_plan = build_material_delivery_plan(selected_content, reply_plan.should_send_material)
        final_reply = FinalReply(
            text=final_text,
            delay_seconds=delay_seconds,
            used_material_ids=selected_material_ids,
            metadata={
                "selected_package_ids": selected_package_ids,
                "selected_material_summary": selected_content.get("prompt_summary"),
                "delivery_plan_summary": delivery_plan.get("summary"),
            },
        )

        if reply_plan.should_reply and final_reply.text.strip():
            self.sender_service.send_text_reply(
                conversation_id,
                final_reply.text,
                final_reply.delay_seconds,
                business_account_id=business_account_id,
                target_chat_id=int((inbound_message.raw_payload or {}).get("chat_id") or 0),
                target_user_id=user_id,
                source_inbound_message_id=inbound_message_id,
                delivery_plan=delivery_plan,
            )
            self.conversation_repo.save_message(
                conversation_id,
                "ai",
                "text",
                final_reply.text,
                {
                    "selected_content": selected_content,
                    "selected_package_ids": selected_package_ids,
                    "used_material_ids": selected_material_ids,
                    "delivery_plan": delivery_plan,
                    "reply_metadata": final_reply.metadata,
                },
                None,
            )
            self.conversation_repo.set_last_ai_reply_at(conversation_id)

        followup_decision = self.proactive_followup_engine.decide(runtime_policy, understanding, mode_decision, reply_plan, user_state, intent_decision, context, latest_user_text, ((self.user_repo.get_user_profile(business_account_id, user_id).get('project_state') or {}).get('project_name')), latest_handover_summary, escalation_decision)
        self.pacing_state_repo.upsert_state(conversation_id, followup_decision.pacing_mode, understanding.boundary_signal, followup_decision.scheduled_for if followup_decision.should_schedule else None, followup_decision.snooze_until, 'low' if not understanding.high_intent_signal else 'medium', {
            'followup_reason': followup_decision.reason,
            'followup_kind': followup_decision.followup_kind,
            'reply_goal': reply_plan.goal,
            'work_hours': self.work_hours_engine.describe_window(runtime_policy),
        }, True, bool(reply_plan.should_reply and final_reply.text.strip()))
        if followup_decision.should_schedule and followup_decision.scheduled_for and followup_decision.text:
            scheduled_for = followup_decision.scheduled_for
            if str(runtime_policy.get('proactive_followup_policy') or 'workhours_only') == 'workhours_only' and not self.work_hours_engine.is_in_reply_window(runtime_policy, scheduled_for):
                scheduled_for = self.work_hours_engine.next_open_time(runtime_policy, scheduled_for)
            self.proactive_followup_repo.schedule_plan(business_account_id, conversation_id, user_id, followup_decision.followup_kind, scheduled_for, {
                'plan_version': 1,
                'text': followup_decision.text,
                'pacing_mode': followup_decision.pacing_mode,
                'followup_kind': followup_decision.followup_kind,
                'source': 'orchestrator',
                'work_hours': self.work_hours_engine.describe_window(runtime_policy),
            }, followup_decision.reason, source_message_id=inbound_message_id)

        self.conversation_repo.update_conversation_state(
            conversation_id,
            stage_decision.stage,
            mode_decision.chat_mode,
            understanding.current_mainline_should_continue,
        )

        project_update_result = self.user_repo.update_project_state(
            business_account_id,
            user_id,
            project_decision.project_id,
            project_decision.reason,
            candidate_projects=project_decision.candidate_projects,
            confidence=project_decision.confidence,
            source=project_decision.source,
            status="classified" if project_decision.project_id is not None else "candidate_only",
            updated_by="system",
        )
        segment_update_result = self.user_repo.update_project_segment_state(
            business_account_id,
            user_id,
            project_decision.project_id,
            segment_decision.get("project_segment_id"),
            segment_decision.get("reason") or "",
            source="ai",
            updated_by="system",
        )
        tag_apply_result = self.user_repo.apply_tag_decision(business_account_id, user_id, tag_decision)

        if ops_decision["changed"]:
            self.user_repo.set_ops_category_manual(
                business_account_id, user_id, ops_decision["ops_category"], ops_decision["reason"], "system"
            )

        if project_update_result.get("updated") and (
            project_decision.project_id is not None or (project_decision.candidate_projects or [])
        ):
            self.receipt_repo.create_system_receipt(
                business_account_id,
                user_id,
                "project_decision",
                "项目判断更新",
                project_decision.reason or "project decision updated",
                {
                    "project_id": project_decision.project_id,
                    "candidate_projects": project_decision.candidate_projects,
                    "confidence": project_decision.confidence,
                    "source": project_decision.source,
                    "segment_name": segment_decision.get("segment_name"),
                    "segment_reason": segment_decision.get("reason"),
                },
            )

        if (tag_apply_result.get("added") or []) or (tag_apply_result.get("removed") or []):
            self.receipt_repo.create_system_receipt(
                business_account_id,
                user_id,
                "tag_decision",
                "标签判断更新",
                tag_decision.reason or "tag decision updated",
                {
                    "added": tag_apply_result.get("added") or [],
                    "removed": tag_apply_result.get("removed") or [],
                    "skipped_locked": tag_apply_result.get("skipped_locked") or [],
                },
            )

        memory_state_snapshot = {
            "stage": stage_decision.stage,
            "chat_mode": mode_decision.chat_mode,
            "ops_category": ops_decision.get("ops_category") or user_state.ops_category,
            "project_id": project_decision.project_id,
            "project_name": ((self.user_repo.get_user_profile(business_account_id, user_id).get("project_state") or {}).get("project_name")),
            "segment_name": segment_decision.get("segment_name"),
            "tags": self.user_repo.get_user_profile(business_account_id, user_id).get("tags") or [],
            "reply_goal": reply_plan.goal,
            "intent_level": intent_decision.level,
            "intent_score": intent_decision.score,
            "tag_changes_added": tag_apply_result.get("added") or [],
            "tag_changes_removed": tag_apply_result.get("removed") or [],
            "project_update_result": project_update_result,
            "segment_update_result": segment_update_result,
            "followup_kind": followup_decision.followup_kind,
            "followup_reason": followup_decision.reason,
            "followup_scheduled_for": followup_decision.scheduled_for.isoformat() if followup_decision.scheduled_for else None,
            "pacing_mode": followup_decision.pacing_mode,
        }
        self.memory_manager.refresh_summary(
            conversation_id,
            context,
            latest_user_text,
            understanding,
            memory_state_snapshot,
            latest_reply=final_reply.text if reply_plan.should_reply and final_reply.text.strip() else None,
            force=bool(project_decision.changed or stage_decision.changed or mode_decision.changed or understanding.high_intent_signal),
        )

        if escalation_decision.get("should_queue_admin"):
            queue_type = "urgent_handover" if escalation_decision.get("notify_level") == "urgent_takeover" else "high_intent"
            priority_score = 95.0 if queue_type == "urgent_handover" else (80.0 if escalation_decision.get("notify_level") == "suggest_takeover" else 60.0)
            self.admin_queue_repo.upsert_queue_item(business_account_id, user_id, queue_type, priority_score, escalation_decision["reason"])
            self.receipt_repo.create_high_intent_receipt(
                business_account_id,
                user_id,
                "检测到高意向用户",
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
    DEFAULT_OPS_CATEGORIES = [
        "new_user",
        "followup_user",
        "high_intent_user",
        "archived_user",
        "invalid_user",
        "blacklist_user",
    ]

    def __init__(self, user_repo: UserRepository, receipt_repo: ReceiptRepository, handover_repo: HandoverRepository,
                 conversation_repo: ConversationRepository, user_control_repo: UserControlRepository,
                 admin_queue_repo: AdminQueueRepository, customer_actions: CustomerActions,
                 resume_chat_manager: ResumeChatManager, project_repo: ProjectRepository | None = None,
                 script_repo: ScriptRepository | None = None,
                 admin_action_receipt_repo: AdminActionReceiptRepository | None = None, material_repo: MaterialRepository | None = None) -> None:
        self.user_repo = user_repo
        self.receipt_repo = receipt_repo
        self.handover_repo = handover_repo
        self.conversation_repo = conversation_repo
        self.user_control_repo = user_control_repo
        self.admin_queue_repo = admin_queue_repo
        self.customer_actions = customer_actions
        self.resume_chat_manager = resume_chat_manager
        self.project_repo = project_repo
        self.script_repo = script_repo
        self.admin_action_receipt_repo = admin_action_receipt_repo
        self.material_repo = material_repo

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
        admin_actions = self.list_recent_admin_actions(business_account_id, user_id, 5)
        resume_decision = None
        if conversation_id and context_payload and context_payload.get("manual_takeover_status") == "pending_resume":
            resume_decision = self.resume_chat_manager.get_resume_decision(conversation_id).__dict__
        project_id_for_materials = None
        try:
            project_id_for_materials = int((profile.get("project_state") or {}).get("project_id") or 0) or None
        except Exception:
            project_id_for_materials = None
        material_library_overview = self.material_repo.get_material_library_overview(business_account_id, project_id_for_materials) if self.material_repo else {}
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
            "admin_action_receipts": admin_actions,
            "material_library_overview": material_library_overview,
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

    def list_projects_admin(self, business_account_id: int, include_inactive: bool = True) -> list[dict]:
        return [] if not self.project_repo else self.project_repo.list_projects_admin(business_account_id, include_inactive)

    def create_project(self, business_account_id: int, name: str, description: str | None = None) -> dict:
        if not self.project_repo:
            return {"ok": False, "reason": "project repository unavailable"}
        row = self.project_repo.create_project(business_account_id, name, description)
        return {"ok": True, "project": row}

    def toggle_project_active(self, project_id: int, is_active: bool) -> dict:
        if not self.project_repo:
            return {"ok": False, "reason": "project repository unavailable"}
        return {"ok": bool(self.project_repo.update_project_active(project_id, is_active))}

    def create_project_segment(self, project_id: int, name: str, description: str | None = None, sort_order: int = 0) -> dict:
        if not self.project_repo:
            return {"ok": False, "reason": "project repository unavailable"}
        row = self.project_repo.create_project_segment(project_id, name, description, sort_order)
        return {"ok": True, "segment": row}

    def list_project_segments_admin(self, project_id: int, include_inactive: bool = True) -> list[dict]:
        return [] if not self.project_repo else self.project_repo.list_project_segments_admin(project_id, include_inactive)

    def toggle_project_segment_active(self, segment_id: int, is_active: bool) -> dict:
        if not self.project_repo:
            return {"ok": False, "reason": "project repository unavailable"}
        return {"ok": bool(self.project_repo.update_project_segment_active(segment_id, is_active))}

    def list_marketing_entries(self, business_account_id: int, project_id: int | None = None, include_inactive: bool = True) -> list[dict]:
        return [] if not self.script_repo else self.script_repo.list_project_scripts_admin(business_account_id, project_id, include_inactive)

    def create_marketing_entry(self, project_id: int, category: str, content_text: str, priority: int = 100) -> dict:
        if not self.script_repo:
            return {"ok": False, "reason": "script repository unavailable"}
        row = self.script_repo.create_project_script(project_id, category, content_text, priority)
        return {"ok": True, "entry": row}

    def toggle_marketing_entry_active(self, entry_id: int, is_active: bool) -> dict:
        if not self.script_repo:
            return {"ok": False, "reason": "script repository unavailable"}
        return {"ok": bool(self.script_repo.update_project_script_active(entry_id, is_active))}

    def list_persona_materials_admin(self, business_account_id: int, include_inactive: bool = True) -> list[dict]:
        return [] if not self.material_repo else self.material_repo.list_persona_materials_admin(business_account_id, include_inactive)

    def create_persona_material(self, business_account_id: int, material_type: str, content_text: str,
                                media_url: str | None = None, scene_tags: list[str] | None = None,
                                priority: int = 100) -> dict:
        if not self.material_repo:
            return {"ok": False, "reason": "material repository unavailable"}
        row = self.material_repo.create_persona_material(business_account_id, material_type, content_text, media_url, scene_tags, priority)
        return {"ok": True, "material": row}

    def toggle_persona_material_active(self, material_id: int, is_active: bool) -> dict:
        if not self.material_repo:
            return {"ok": False, "reason": "material repository unavailable"}
        return {"ok": bool(self.material_repo.update_material_active("persona_materials", material_id, is_active))}

    def list_daily_materials_admin(self, business_account_id: int, include_inactive: bool = True) -> list[dict]:
        return [] if not self.material_repo else self.material_repo.list_daily_materials_admin(business_account_id, include_inactive)

    def create_daily_material(self, business_account_id: int, category: str, content_text: str,
                              media_url: str | None = None, scene_tags: list[str] | None = None,
                              priority: int = 100) -> dict:
        if not self.material_repo:
            return {"ok": False, "reason": "material repository unavailable"}
        row = self.material_repo.create_daily_material(business_account_id, category, content_text, media_url, scene_tags, priority)
        return {"ok": True, "material": row}

    def toggle_daily_material_active(self, material_id: int, is_active: bool) -> dict:
        if not self.material_repo:
            return {"ok": False, "reason": "material repository unavailable"}
        return {"ok": bool(self.material_repo.update_material_active("daily_materials", material_id, is_active))}

    def list_project_materials_admin(self, project_id: int, include_inactive: bool = True) -> list[dict]:
        return [] if not self.material_repo else self.material_repo.list_project_materials_admin(project_id, include_inactive)

    def create_project_material(self, project_id: int, material_type: str, content_text: str,
                                media_url: str | None = None, scene_tags: list[str] | None = None,
                                priority: int = 100) -> dict:
        if not self.material_repo:
            return {"ok": False, "reason": "material repository unavailable"}
        row = self.material_repo.create_project_material(project_id, material_type, content_text, media_url, scene_tags, priority)
        return {"ok": True, "material": row}

    def toggle_project_material_active(self, material_id: int, is_active: bool) -> dict:
        if not self.material_repo:
            return {"ok": False, "reason": "material repository unavailable"}
        return {"ok": bool(self.material_repo.update_material_active("project_materials", material_id, is_active))}

    def list_all_tag_names(self, business_account_id: int) -> list[str]:
        db = self.conversation_repo.db
        with db.cursor() as cur:
            cur.execute("SELECT name FROM tags WHERE business_account_id=%s AND is_active=TRUE ORDER BY name ASC", (business_account_id,))
            rows = cur.fetchall()
        return [r["name"] for r in rows]

    def list_active_tag_names_for_user(self, business_account_id: int, user_id: int) -> list[str]:
        detail = self.get_customer_detail(business_account_id, user_id)
        return [] if not detail.get("ok") else ((detail.get("profile", {}) or {}).get("tags", []) or [])

    def list_available_ops_categories(self) -> list[str]:
        return list(self.DEFAULT_OPS_CATEGORIES)

    def list_recent_admin_actions(self, business_account_id: int, user_id: int, limit: int = 10) -> list[dict]:
        if not self.admin_action_receipt_repo:
            return []
        return self.admin_action_receipt_repo.list_recent_by_user(business_account_id, user_id, limit)

    def record_admin_action(self, business_account_id: int | None, admin_chat_id: int, action_type: str,
                            message_text: str, target_user_id: int | None = None,
                            target_project_id: int | None = None, payload: dict[str, Any] | None = None,
                            result_status: str = 'success') -> None:
        if not self.admin_action_receipt_repo:
            return
        self.admin_action_receipt_repo.create_receipt(
            business_account_id=business_account_id,
            admin_chat_id=admin_chat_id,
            action_type=action_type,
            message_text=message_text,
            target_user_id=target_user_id,
            target_project_id=target_project_id,
            payload=payload,
            result_status=result_status,
        )

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
    def _short_time(value: Any) -> str:
        if isinstance(value, datetime):
            try:
                return value.astimezone(timezone.utc).strftime("%m-%d %H:%M")
            except Exception:
                return value.strftime("%m-%d %H:%M")
        return "-"

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
            f"- 高意向用户: {ops.get('high_intent_user', 0)}\n"
            f"- 已归档: {ops.get('archived_user', 0)}\n"
            f"- 无效用户: {ops.get('invalid_user', 0)}\n"
            f"- 黑名单: {ops.get('blacklist_user', 0)}"
        )

    @staticmethod
    def format_queue_list(title: str, items: list[dict]) -> str:
        if not items:
            return f"{title}\n\n暂无数据。"
        lines = [title, ""]
        for idx, item in enumerate(items[:10], start=1):
            lines.append(
                f"{idx}. 用户ID: {item.get('user_id')} | 类型: {item.get('queue_type', '-') or '-'} | 优先级: {item.get('priority_score', '-') or '-'} | 原因: {(item.get('reason_text') or '-')[:70]}"
            )
        lines.append("\n点击下方按钮可查看客户详情。")
        return "\n".join(lines)

    @staticmethod
    def format_pending_resume_list(items: list[dict]) -> str:
        if not items:
            return "⏳ 待恢复AI列表\n\n暂无待恢复会话。"
        lines = ["⏳ 待恢复AI列表", ""]
        for idx, item in enumerate(items[:10], start=1):
            lines.append(f"{idx}. 会话ID: {item.get('conversation_id')} | 用户ID: {item.get('user_id')} | 项目: {item.get('project_name') or '-'}")
        lines.append("\n点击下方按钮可恢复AI或查看详情。")
        return "\n".join(lines)

    @staticmethod
    def format_project_selection(projects: list[dict], current_project_id: int | None) -> str:
        lines = ["🧭 手动设置项目", "", f"当前项目ID: {current_project_id or '-'}", "", "请输入项目ID。输入 0 / none / clear 可清空项目。", ""]
        if not projects:
            lines.append("当前没有可用项目。")
            return "\n".join(lines)
        lines.append("可用项目列表:")
        for item in projects[:20]:
            lines.append(f"- {item.get('id')}: {item.get('name') or '-'}")
        return "\n".join(lines)

    @staticmethod
    def format_tag_prompt(title: str, tags: list[str], active_tags: list[str] | None = None) -> str:
        lines = [title, ""]
        if active_tags is not None:
            lines.append(f"当前标签: {', '.join(active_tags) if active_tags else '-'}")
            lines.append("")
        lines.append("支持逗号、空格、换行混合输入多个标签。")
        lines.append("")
        if tags:
            lines.append("可用标签:")
            lines.append(", ".join(tags[:50]))
        else:
            lines.append("当前没有可用标签。")
        return "\n".join(lines)

    @staticmethod
    def format_ops_category_prompt(current_ops: str | None) -> str:
        return (
            "📦 手动设置运营分类\n\n"
            f"当前分类: {current_ops or '-'}\n\n"
            "请点击下方按钮选择新的运营分类。"
        )

    @staticmethod
    @staticmethod
    def format_customer_detail(detail: dict) -> str:
        if not detail.get("ok"):
            return "❌ 未找到客户。"
        profile = detail.get("profile") or {}
        user_row = profile.get("user") or {}
        ops = profile.get("ops_status") or {}
        project = profile.get("project_state") or {}
        segment = profile.get("segment_state") or {}
        tags = profile.get("tags") or []
        conv = detail.get("conversation") or {}
        ctx = conv.get("context") or {}
        summary = detail.get("latest_handover_summary") or {}
        ai_control = detail.get("ai_control") or {}
        admin_actions = detail.get("admin_action_receipts") or []
        long_term_memory = (ctx.get("long_term_memory") or {}) if isinstance(ctx, dict) else {}
        candidate_projects = project.get("candidate_projects_json") or []
        lines = [
            "👤 客户详情",
            "",
            f"用户ID: {user_row.get('id', '-')}",
            f"TG用户ID: {user_row.get('tg_user_id', '-')}",
            f"昵称: {user_row.get('display_name') or '-'}",
            f"用户名: {user_row.get('username') or '-'}",
            f"运营分类: {ops.get('ops_category') or '-'} | 来源: {ops.get('source') or '-'} | 锁定: {'是' if ops.get('is_locked') else '否'}",
            f"项目: {project.get('project_name') or '-'} | 来源: {project.get('source') or '-'} | 锁定: {'是' if project.get('is_locked') else '否'} | 置信度: {project.get('confidence') if project.get('confidence') is not None else '-'}",
            f"阶段: {segment.get('segment_name') or '-'}",
            f"标签: {', '.join(tags) if tags else '-'}",
            f"会话ID: {conv.get('conversation_id') or '-'}",
            f"当前模式: {ctx.get('current_chat_mode') or '-'}",
            f"当前阶段: {ctx.get('current_stage') or '-'}",
            f"接管状态: {ctx.get('manual_takeover_status') or '-'}",
            f"AI开关覆盖: {ai_control.get('ai_enabled_override') if ai_control else '-'}",
        ]
        if long_term_memory:
            lines.extend([
                "",
                f"滚动记忆: {(long_term_memory.get('summary_text') or '-')[:160]}",
            ])
            boundaries = long_term_memory.get("boundaries") or []
            if boundaries:
                lines.append(f"边界/节奏: {', '.join(str(i) for i in boundaries[:4])}")
            followup = long_term_memory.get("followup_strategy")
            if followup:
                lines.append(f"跟进策略: {str(followup)[:120]}")
        if candidate_projects:
            top_candidates = []
            for item in candidate_projects[:3]:
                try:
                    top_candidates.append(f"{item.get('project_name') or item.get('project_id')}: {item.get('score')}")
                except Exception:
                    continue
            if top_candidates:
                lines.extend(["", f"候选项目: {' | '.join(top_candidates)}"])
        material_overview = detail.get("material_library_overview") or {}
        if material_overview:
            lines.extend([
                "",
                f"素材库: 人设{material_overview.get('persona_material_count', 0)} | 日常{material_overview.get('daily_material_count', 0)} | 项目{material_overview.get('project_material_count', 0)}",
                f"素材包: 人设{material_overview.get('persona_package_count', 0)} | 日常{material_overview.get('daily_package_count', 0)} | 项目{material_overview.get('project_package_count', 0)}",
            ])
        if summary:
            lines.extend(["", f"最近接管摘要: {(summary.get('theme_summary') or '-')[:120]}"])
        if admin_actions:
            lines.extend(["", "最近管理员操作:"])
            for item in admin_actions[:3]:
                lines.append(f"- {TGAdminFormatter._short_time(item.get('created_at'))} | {item.get('action_type') or '-'} | {(item.get('message_text') or '-')[:80]}")
        return "\n".join(lines)

    @staticmethod
    def format_project_admin_list(projects: list[dict]) -> str:
        if not projects:
            return "🧱 项目管理\n\n暂无项目。"
        lines = ["🧱 项目管理", "", "当前项目列表："]
        for item in projects[:20]:
            status = "启用" if item.get("is_active") else "停用"
            lines.append(f"- {item.get('id')}: {item.get('name') or '-'} | {status} | {(item.get('description') or '-')[:48]}")
        lines.append("")
        lines.append("可新建项目，也可点项目按钮查看详情/管理阶段。")
        return "\n".join(lines)

    @staticmethod
    def format_project_segment_admin(project_row: dict | None, segments: list[dict]) -> str:
        if not project_row:
            return "❌ 项目不存在。"
        lines = [f"🧭 项目阶段管理 | {project_row.get('name') or '-'}", "", f"项目ID: {project_row.get('id')}"]
        if not segments:
            lines.extend(["", "暂无阶段。"])
        else:
            lines.extend(["", "阶段列表："])
            for item in segments[:20]:
                status = "启用" if item.get("is_active") else "停用"
                lines.append(f"- {item.get('id')}: {item.get('name') or '-'} | 顺序 {item.get('sort_order', 0)} | {status}")
        lines.append("")
        lines.append("可新建阶段，或切换阶段启停。")
        return "\n".join(lines)

    @staticmethod
    def format_marketing_entry_list(entries: list[dict]) -> str:
        if not entries:
            return "📝 客户营销方案 / 文案库\n\n暂无内容。"
        lines = ["📝 客户营销方案 / 文案库", "", "当前条目："]
        for item in entries[:20]:
            status = "启用" if item.get("is_active") else "停用"
            lines.append(f"- {item.get('id')}: 项目 {item.get('project_name') or item.get('project_id')} | 分类 {item.get('category') or '-'} | 优先级 {item.get('priority', '-')} | {status}")
            lines.append(f"  {(item.get('content_text') or '-')[:88]}")
        return "\n".join(lines)

    @staticmethod
    def format_material_admin_list(title: str, items: list[dict], kind_label: str) -> str:
        if not items:
            return f"{title}\n\n暂无内容。"
        lines = [title, "", "当前素材："]
        for item in items[:20]:
            status = "启用" if item.get("is_active") else "停用"
            scene_tags = item.get("scene_tags_json") or []
            scene_text = ", ".join(str(v) for v in scene_tags[:3]) if scene_tags else "-"
            type_value = item.get("material_type") or item.get("category") or kind_label
            lines.append(f"- {item.get('id')}: 类型 {type_value} | 优先级 {item.get('priority', '-')} | {status} | 标签 {scene_text}")
            if item.get('content_text'):
                lines.append(f"  {(item.get('content_text') or '-')[:88]}")
            if item.get('media_url'):
                lines.append(f"  媒体: {(item.get('media_url') or '-')[:88]}")
        return "\n".join(lines)

    @staticmethod
    def format_content_create_prompt(title: str, template_lines: list[str]) -> str:
        return "\n".join([title, "", "请输入以下格式：", *template_lines])


class TGAdminMenuBuilder:
    @staticmethod
    def main_menu() -> dict:
        return {
            "text": "📊 管理后台\n请选择操作：",
            "reply_markup": [
                [{"text": "📈 仪表盘", "callback_data": "adm:dashboard"}, {"text": "🔥 高意向客户", "callback_data": "adm:queue:high_intent"}],
                [{"text": "⚠️ 紧急处理", "callback_data": "adm:queue:urgent_handover"}, {"text": "⏳ 待恢复AI", "callback_data": "adm:queue:pending_resume"}],
                [{"text": "📂 打开队列", "callback_data": "adm:queue:all"}, {"text": "🔎 搜索客户", "callback_data": "adm:search"}],
                [{"text": "🧱 内容管理", "callback_data": "adm:content"}],
            ],
        }

    @staticmethod
    def content_menu() -> dict:
        return {
            "text": "🧱 内容管理\n请选择要维护的板块：",
            "reply_markup": [
                [{"text": "📁 项目管理", "callback_data": "adm:content:projects"}, {"text": "📝 营销方案库", "callback_data": "adm:content:marketing"}],
                [{"text": "👩 人设素材", "callback_data": "adm:content:persona"}, {"text": "🌤️ 日常素材", "callback_data": "adm:content:daily"}],
                [{"text": "📦 项目素材", "callback_data": "adm:content:project_materials"}],
                [{"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ],
        }

    @staticmethod
    def content_section_menu(section: str) -> list[list[dict[str, str]]]:
        mapping = {
            "projects": [
                [{"text": "➕ 新建项目", "callback_data": "adm:content:new_project"}, {"text": "🧭 新建阶段", "callback_data": "adm:content:segments"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ],
            "marketing": [
                [{"text": "➕ 新建营销方案", "callback_data": "adm:content:new_marketing"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ],
            "persona": [
                [{"text": "➕ 新建人设素材", "callback_data": "adm:content:new_persona_material"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ],
            "daily": [
                [{"text": "➕ 新建日常素材", "callback_data": "adm:content:new_daily_material"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ],
            "project_materials": [
                [{"text": "➕ 新建项目素材", "callback_data": "adm:content:new_project_material"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ],
        }
        return mapping.get(section, [[{"text": "🧱 返回内容管理", "callback_data": "adm:content"}]])

    @staticmethod
    def customer_action_menu(user_id: int, conversation_id: int | None, back_callback: str = "adm:main") -> list[list[dict[str, str]]]:
        rows: list[list[dict[str, str]]] = [
            [
                {"text": "🧭 改项目", "callback_data": f"adm:project:{user_id}:{back_callback}"},
                {"text": "🏷️ 加标签", "callback_data": f"adm:tagadd:{user_id}:{back_callback}"},
            ],
            [
                {"text": "🗑️ 删标签", "callback_data": f"adm:tagremove:{user_id}:{back_callback}"},
                {"text": "📦 运营分类", "callback_data": f"adm:opsmenu:{user_id}:{back_callback}"},
            ],
        ]
        if conversation_id:
            rows.append([
                {"text": "🧑 人工接管", "callback_data": f"adm:takeover:{user_id}:{conversation_id}:{back_callback}"},
                {"text": "🤖 恢复AI", "callback_data": f"adm:resume:{conversation_id}:{user_id}:{back_callback}"},
            ])
            rows.append([
                {"text": "🛑 停用AI", "callback_data": f"adm:ai:disable:{conversation_id}:{user_id}:{back_callback}"},
                {"text": "✅ 启用AI", "callback_data": f"adm:ai:enable:{conversation_id}:{user_id}:{back_callback}"},
            ])
        rows.append([
            {"text": "🔙 返回", "callback_data": back_callback},
            {"text": "🏠 主菜单", "callback_data": "adm:main"},
        ])
        return rows

    @staticmethod
    def queue_list_menu(items: list[dict], back_callback: str) -> list[list[dict[str, str]]]:
        rows: list[list[dict[str, str]]] = []
        for item in items[:10]:
            rows.append([{"text": f"查看用户 {item.get('user_id')}", "callback_data": f"adm:customer:{item.get('user_id')}:{back_callback}"}])
        rows.append([{"text": "🔙 返回主菜单", "callback_data": "adm:main"}])
        return rows

    @staticmethod
    def pending_resume_menu(items: list[dict], back_callback: str) -> list[list[dict[str, str]]]:
        rows: list[list[dict[str, str]]] = []
        for item in items[:10]:
            rows.append([
                {"text": f"用户 {item.get('user_id')} 详情", "callback_data": f"adm:customer:{item.get('user_id')}:{back_callback}"},
                {"text": f"恢复AI {item.get('conversation_id')}", "callback_data": f"adm:resume:{item.get('conversation_id')}:{item.get('user_id')}:{back_callback}"},
            ])
        rows.append([{"text": "🔙 返回主菜单", "callback_data": "adm:main"}])
        return rows

    @staticmethod
    def ops_category_menu(user_id: int, back_callback: str) -> list[list[dict[str, str]]]:
        return [
            [
                {"text": "新用户", "callback_data": f"adm:opsset:{user_id}:new_user:{back_callback}"},
                {"text": "跟进用户", "callback_data": f"adm:opsset:{user_id}:followup_user:{back_callback}"},
            ],
            [
                {"text": "高意向", "callback_data": f"adm:opsset:{user_id}:high_intent_user:{back_callback}"},
                {"text": "已归档", "callback_data": f"adm:opsset:{user_id}:archived_user:{back_callback}"},
            ],
            [
                {"text": "无效用户", "callback_data": f"adm:opsset:{user_id}:invalid_user:{back_callback}"},
                {"text": "黑名单", "callback_data": f"adm:opsset:{user_id}:blacklist_user:{back_callback}"},
            ],
            [
                {"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"},
                {"text": "🏠 主菜单", "callback_data": "adm:main"},
            ],
        ]


class TGAdminCallbackRouter:
    def __init__(self, admin_api_service: AdminAPIService, dashboard_service: DashboardService, tg_sender,
                 business_account_repo: BusinessAccountRepository, admin_session_repo: AdminSessionRepository) -> None:
        self.admin_api_service = admin_api_service
        self.dashboard_service = dashboard_service
        self.tg_sender = tg_sender
        self.business_account_repo = business_account_repo
        self.admin_session_repo = admin_session_repo

    def _send_text(self, admin_chat_id: int, text: str, reply_markup=None) -> None:
        if reply_markup:
            self.tg_sender.send_admin_message(admin_chat_id, text, reply_markup)
        else:
            self.tg_sender.send_admin_text(chat_id=admin_chat_id, text=text)

    def _resolve_business_account_id(self) -> int | None:
        return self.business_account_repo.get_default_account_id()

    @staticmethod
    def _tail(parts: list[str], start_index: int, default: str = "adm:main") -> str:
        return ":".join(parts[start_index:]) if len(parts) > start_index else default

    def _record_action(self, business_account_id: int | None, admin_chat_id: int, action_type: str, message_text: str,
                       target_user_id: int | None = None, target_project_id: int | None = None,
                       payload: dict[str, Any] | None = None, result_status: str = 'success') -> None:
        self.admin_api_service.record_admin_action(
            business_account_id=business_account_id,
            admin_chat_id=admin_chat_id,
            action_type=action_type,
            message_text=message_text,
            target_user_id=target_user_id,
            target_project_id=target_project_id,
            payload=payload,
            result_status=result_status,
        )

    def _activate_text_input(self, admin_chat_id: int, business_account_id: int, state: str,
                             prompt_text: str, payload: dict[str, Any] | None = None,
                             reply_markup=None) -> None:
        self.admin_session_repo.set_session(
            admin_chat_id,
            state,
            {"business_account_id": business_account_id, **(payload or {})},
            business_account_id=business_account_id,
            expires_at=utc_now() + timedelta(minutes=15),
        )
        self._send_text(admin_chat_id, prompt_text, reply_markup)

    def handle(self, admin_chat_id: int, callback_data: str, operator: str = "admin") -> None:
        parts = (callback_data or "").split(":")
        if not parts or parts[0] != "adm":
            self._send_text(admin_chat_id, "❌ 未知的管理员回调。")
            return
        business_account_id = self._resolve_business_account_id()
        action = parts[1] if len(parts) > 1 else "main"
        if action == "main":
            self.admin_session_repo.clear_session(admin_chat_id)
            menu = TGAdminMenuBuilder.main_menu()
            self._send_text(admin_chat_id, menu["text"], menu["reply_markup"])
            return
        if business_account_id is None:
            self._send_text(admin_chat_id, "⚠️ 当前还没有可用业务账号，请先让机器人收到至少一条业务消息。")
            return
        if action == "dashboard":
            summary = self.dashboard_service.get_summary(business_account_id)
            self._send_text(admin_chat_id, TGAdminFormatter.format_dashboard(summary), [[{"text": "🔙 返回主菜单", "callback_data": "adm:main"}]])
            return
        if action == "search":
            self.admin_session_repo.set_session(admin_chat_id, "search_customer", {"business_account_id": business_account_id}, business_account_id=business_account_id, expires_at=utc_now() + timedelta(minutes=10))
            self._send_text(admin_chat_id, "🔎 请输入客户关键词、TG用户ID、用户名或昵称。")
            return
        if action == "queue":
            queue_type = parts[2] if len(parts) > 2 else "all"
            if queue_type == "pending_resume":
                items = self.admin_api_service.list_pending_resume_items(business_account_id)
                self._send_text(admin_chat_id, TGAdminFormatter.format_pending_resume_list(items), TGAdminMenuBuilder.pending_resume_menu(items, f"adm:queue:{queue_type}"))
                return
            normalized_queue_type = None if queue_type == "all" else queue_type
            items = self.dashboard_service.list_open_queue(business_account_id, normalized_queue_type, 20)
            title_map = {"all": "📂 打开队列", "high_intent": "🔥 高意向客户", "urgent_handover": "⚠️ 紧急处理"}
            title = title_map.get(queue_type, "📂 队列")
            self._send_text(admin_chat_id, TGAdminFormatter.format_queue_list(title, items), TGAdminMenuBuilder.queue_list_menu(items, f"adm:queue:{queue_type}"))
            return
        if action == "customer" and len(parts) >= 3:
            user_id = int(parts[2])
            back_callback = self._tail(parts, 3)
            detail = self.admin_api_service.get_customer_detail(business_account_id, user_id)
            conversation_id = ((detail.get("conversation") or {}).get("conversation_id") if detail.get("ok") else None)
            self._send_text(admin_chat_id, TGAdminFormatter.format_customer_detail(detail), TGAdminMenuBuilder.customer_action_menu(user_id, conversation_id, back_callback))
            return
        if action == "project" and len(parts) >= 3:
            user_id = int(parts[2])
            back_callback = self._tail(parts, 3)
            detail = self.admin_api_service.get_customer_detail(business_account_id, user_id)
            current_project_id = ((detail.get("profile") or {}).get("project_state") or {}).get("project_id") if detail.get("ok") else None
            projects = self.admin_api_service.list_projects_for_business_account(business_account_id)
            self.admin_session_repo.set_session(
                admin_chat_id,
                "set_project_target",
                {
                    "business_account_id": business_account_id,
                    "user_id": user_id,
                    "back_callback": back_callback,
                },
                business_account_id=business_account_id,
                expires_at=utc_now() + timedelta(minutes=10),
            )
            self._send_text(admin_chat_id, TGAdminFormatter.format_project_selection(projects, current_project_id), [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}]])
            return
        if action == "tagadd" and len(parts) >= 3:
            user_id = int(parts[2])
            back_callback = self._tail(parts, 3)
            tags = self.admin_api_service.list_all_tag_names(business_account_id)
            active_tags = self.admin_api_service.list_active_tag_names_for_user(business_account_id, user_id)
            self.admin_session_repo.set_session(
                admin_chat_id,
                "add_tag_target",
                {"business_account_id": business_account_id, "user_id": user_id, "back_callback": back_callback},
                business_account_id=business_account_id,
                expires_at=utc_now() + timedelta(minutes=10),
            )
            self._send_text(admin_chat_id, TGAdminFormatter.format_tag_prompt("🏷️ 手动添加标签", tags, active_tags), [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}]])
            return
        if action == "tagremove" and len(parts) >= 3:
            user_id = int(parts[2])
            back_callback = self._tail(parts, 3)
            active_tags = self.admin_api_service.list_active_tag_names_for_user(business_account_id, user_id)
            self.admin_session_repo.set_session(
                admin_chat_id,
                "remove_tag_target",
                {"business_account_id": business_account_id, "user_id": user_id, "back_callback": back_callback},
                business_account_id=business_account_id,
                expires_at=utc_now() + timedelta(minutes=10),
            )
            self._send_text(admin_chat_id, TGAdminFormatter.format_tag_prompt("🗑️ 手动移除标签", active_tags, active_tags), [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}]])
            return
        if action == "opsmenu" and len(parts) >= 3:
            user_id = int(parts[2])
            back_callback = self._tail(parts, 3)
            detail = self.admin_api_service.get_customer_detail(business_account_id, user_id)
            current_ops = ((detail.get("profile") or {}).get("ops_status") or {}).get("ops_category") if detail.get("ok") else None
            self._send_text(admin_chat_id, TGAdminFormatter.format_ops_category_prompt(current_ops), TGAdminMenuBuilder.ops_category_menu(user_id, back_callback))
            return
        if action == "opsset" and len(parts) >= 4:
            user_id = int(parts[2])
            ops_category = parts[3]
            back_callback = self._tail(parts, 4)
            result = self.admin_api_service.set_ops_category(business_account_id, user_id, ops_category, operator, reason_text="telegram admin manual ops update")
            message = f"✅ 已更新运营分类为 {ops_category}。" if result.get("ok") else f"❌ 更新运营分类失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "set_ops_category", message, target_user_id=user_id, payload={"ops_category": ops_category}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            return
        if action == "takeover" and len(parts) >= 4:
            user_id = int(parts[2])
            conversation_id = int(parts[3])
            back_callback = self._tail(parts, 4)
            result = self.admin_api_service.start_manual_handover(business_account_id, user_id, conversation_id, operator, reason="telegram admin manual takeover")
            message = "✅ 已开始人工接管。" if result.get("ok") else f"❌ 接管失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "manual_takeover", message, target_user_id=user_id, payload={"conversation_id": conversation_id}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            return
        if action == "resume" and len(parts) >= 3:
            conversation_id = int(parts[2])
            user_id = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else None
            back_callback = self._tail(parts, 4)
            result = self.admin_api_service.resume_ai(conversation_id, operator)
            message = "✅ 已恢复 AI 自动聊天。" if result.get("ok") else f"❌ 恢复失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "resume_ai", message, target_user_id=user_id, payload={"conversation_id": conversation_id}, result_status='success' if result.get('ok') else 'failed')
            receipt_buttons = []
            if user_id is not None:
                receipt_buttons.append([{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}])
            receipt_buttons.append([{"text": "🏠 主菜单", "callback_data": "adm:main"}])
            self._send_text(admin_chat_id, message, receipt_buttons)
            return
        if action == "ai" and len(parts) >= 5:
            mode = parts[2]
            conversation_id = int(parts[3])
            user_id = int(parts[4]) if parts[4].isdigit() else None
            back_callback = self._tail(parts, 5)
            if mode == "disable":
                result = self.admin_api_service.disable_ai_for_conversation(conversation_id, operator)
                message = "✅ 已停用该会话 AI 自动回复。" if result.get("ok") else f"❌ 停用AI失败：{result.get('reason') or '未知错误'}"
                action_type = "disable_ai"
            else:
                result = self.admin_api_service.enable_ai_for_conversation(conversation_id, operator)
                message = "✅ 已启用该会话 AI 自动回复。" if result.get("ok") else f"❌ 启用AI失败：{result.get('reason') or '未知错误'}"
                action_type = "enable_ai"
            self._record_action(business_account_id, admin_chat_id, action_type, message, target_user_id=user_id, payload={"conversation_id": conversation_id}, result_status='success' if result.get('ok') else 'failed')
            receipt_buttons = []
            if user_id is not None:
                receipt_buttons.append([{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}])
            receipt_buttons.append([{"text": "🏠 主菜单", "callback_data": "adm:main"}])
            self._send_text(admin_chat_id, message, receipt_buttons)
            return
        if action == "content":
            section = parts[2] if len(parts) > 2 else ""
            if not section:
                menu = TGAdminMenuBuilder.content_menu()
                self._send_text(admin_chat_id, menu["text"], menu["reply_markup"])
                return
            if section == "projects":
                projects = self.admin_api_service.list_projects_admin(business_account_id, include_inactive=True)
                buttons: list[list[dict[str, str]]] = []
                for item in projects[:12]:
                    status_action = "停用" if item.get("is_active") else "启用"
                    toggle_flag = "0" if item.get("is_active") else "1"
                    buttons.append([
                        {"text": f"项目 {item.get('id')}", "callback_data": f"adm:content:project_detail:{item.get('id')}"},
                        {"text": status_action, "callback_data": f"adm:content:toggle_project:{item.get('id')}:{toggle_flag}"},
                    ])
                buttons.extend(TGAdminMenuBuilder.content_section_menu("projects"))
                self._send_text(admin_chat_id, TGAdminFormatter.format_project_admin_list(projects), buttons)
                return
            if section == "project_detail" and len(parts) >= 4 and parts[3].isdigit():
                project_id = int(parts[3])
                project_row = next((item for item in self.admin_api_service.list_projects_admin(business_account_id, True) if int(item.get("id")) == project_id), None)
                segments = self.admin_api_service.list_project_segments_admin(project_id, True)
                buttons: list[list[dict[str, str]]] = []
                for segment in segments[:10]:
                    seg_toggle = "0" if segment.get("is_active") else "1"
                    buttons.append([{"text": f"阶段 {segment.get('id')} {'停用' if segment.get('is_active') else '启用'}", "callback_data": f"adm:content:toggle_segment:{segment.get('id')}:{seg_toggle}:project:{project_id}"}])
                buttons.append([{"text": "➕ 新建阶段", "callback_data": f"adm:content:new_segment:{project_id}"}])
                buttons.append([{"text": "🔙 返回项目列表", "callback_data": "adm:content:projects"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}])
                self._send_text(admin_chat_id, TGAdminFormatter.format_project_segment_admin(project_row, segments), buttons)
                return
            if section == "toggle_project" and len(parts) >= 5 and parts[3].isdigit():
                project_id = int(parts[3]); is_active = parts[4] == "1"
                result = self.admin_api_service.toggle_project_active(project_id, is_active)
                message = "✅ 已更新项目状态。" if result.get("ok") else "❌ 更新项目状态失败。"
                self._record_action(business_account_id, admin_chat_id, "toggle_project", message, target_project_id=project_id, payload={"project_id": project_id, "is_active": is_active}, result_status='success' if result.get('ok') else 'failed')
                self._send_text(admin_chat_id, message, [[{"text": "🔙 返回项目列表", "callback_data": "adm:content:projects"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
                return
            if section == "toggle_segment" and len(parts) >= 5 and parts[3].isdigit():
                segment_id = int(parts[3]); is_active = parts[4] == "1"
                project_id = int(parts[6]) if len(parts) >= 7 and parts[5] == "project" and parts[6].isdigit() else None
                result = self.admin_api_service.toggle_project_segment_active(segment_id, is_active)
                message = "✅ 已更新阶段状态。" if result.get("ok") else "❌ 更新阶段状态失败。"
                buttons = [[{"text": "🔙 返回项目阶段", "callback_data": f"adm:content:project_detail:{project_id}"}]] if project_id else [[{"text": "🔙 返回项目列表", "callback_data": "adm:content:projects"}]]
                buttons.append([{"text": "🏠 主菜单", "callback_data": "adm:main"}])
                self._send_text(admin_chat_id, message, buttons)
                return
            if section == "new_project":
                self._activate_text_input(admin_chat_id, business_account_id, "content_new_project", TGAdminFormatter.format_content_create_prompt("📁 新建项目", ["名称 | 描述（可选）"]), {"back_callback": "adm:content:projects"}, [[{"text": "🔙 返回项目列表", "callback_data": "adm:content:projects"}]])
                return
            if section == "segments":
                self._activate_text_input(admin_chat_id, business_account_id, "content_new_segment", TGAdminFormatter.format_content_create_prompt("🧭 新建项目阶段", ["项目ID | 阶段名称 | 描述（可选） | 排序（可选，默认0）"]), {"back_callback": "adm:content:projects"}, [[{"text": "🔙 返回项目列表", "callback_data": "adm:content:projects"}]])
                return
            if section == "new_segment" and len(parts) >= 4 and parts[3].isdigit():
                project_id = int(parts[3])
                self._activate_text_input(admin_chat_id, business_account_id, "content_new_segment", TGAdminFormatter.format_content_create_prompt("🧭 新建项目阶段", [f"{project_id} | 阶段名称 | 描述（可选） | 排序（可选，默认0)"]), {"back_callback": f"adm:content:project_detail:{project_id}", "project_id": project_id}, [[{"text": "🔙 返回项目阶段", "callback_data": f"adm:content:project_detail:{project_id}"}]])
                return
            if section == "marketing":
                entries = self.admin_api_service.list_marketing_entries(business_account_id, include_inactive=True)
                buttons: list[list[dict[str, str]]] = []
                for item in entries[:10]:
                    toggle_flag = "0" if item.get("is_active") else "1"
                    buttons.append([{"text": f"文案 {item.get('id')} {'停用' if item.get('is_active') else '启用'}", "callback_data": f"adm:content:toggle_marketing:{item.get('id')}:{toggle_flag}"}])
                buttons.extend(TGAdminMenuBuilder.content_section_menu("marketing"))
                self._send_text(admin_chat_id, TGAdminFormatter.format_marketing_entry_list(entries), buttons)
                return
            if section == "new_marketing":
                self._activate_text_input(admin_chat_id, business_account_id, "content_new_marketing", TGAdminFormatter.format_content_create_prompt("📝 新建营销方案 / 文案", ["项目ID | 分类 | 优先级（可选，默认100） | 内容"]), {"back_callback": "adm:content:marketing"}, [[{"text": "🔙 返回营销方案库", "callback_data": "adm:content:marketing"}]])
                return
            if section == "toggle_marketing" and len(parts) >= 5 and parts[3].isdigit():
                entry_id = int(parts[3]); is_active = parts[4] == "1"
                result = self.admin_api_service.toggle_marketing_entry_active(entry_id, is_active)
                message = "✅ 已更新营销方案状态。" if result.get("ok") else "❌ 更新营销方案状态失败。"
                self._send_text(admin_chat_id, message, [[{"text": "🔙 返回营销方案库", "callback_data": "adm:content:marketing"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
                return
            if section == "persona":
                items = self.admin_api_service.list_persona_materials_admin(business_account_id, include_inactive=True)
                buttons: list[list[dict[str, str]]] = []
                for item in items[:10]:
                    toggle_flag = "0" if item.get("is_active") else "1"
                    buttons.append([{"text": f"素材 {item.get('id')} {'停用' if item.get('is_active') else '启用'}", "callback_data": f"adm:content:toggle_persona:{item.get('id')}:{toggle_flag}"}])
                buttons.extend(TGAdminMenuBuilder.content_section_menu("persona"))
                self._send_text(admin_chat_id, TGAdminFormatter.format_material_admin_list("👩 人设素材管理", items, "persona"), buttons)
                return
            if section == "new_persona_material":
                self._activate_text_input(admin_chat_id, business_account_id, "content_new_persona_material", TGAdminFormatter.format_content_create_prompt("👩 新建人设素材", ["类型 | 优先级（可选，默认100） | 场景标签逗号（可选） | 文本内容", "如果需要媒体链接，可在末尾继续追加 | 媒体URL"]), {"back_callback": "adm:content:persona"}, [[{"text": "🔙 返回人设素材", "callback_data": "adm:content:persona"}]])
                return
            if section == "toggle_persona" and len(parts) >= 5 and parts[3].isdigit():
                material_id = int(parts[3]); is_active = parts[4] == "1"
                result = self.admin_api_service.toggle_persona_material_active(material_id, is_active)
                message = "✅ 已更新人设素材状态。" if result.get("ok") else "❌ 更新人设素材状态失败。"
                self._send_text(admin_chat_id, message, [[{"text": "🔙 返回人设素材", "callback_data": "adm:content:persona"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
                return
            if section == "daily":
                items = self.admin_api_service.list_daily_materials_admin(business_account_id, include_inactive=True)
                buttons: list[list[dict[str, str]]] = []
                for item in items[:10]:
                    toggle_flag = "0" if item.get("is_active") else "1"
                    buttons.append([{"text": f"素材 {item.get('id')} {'停用' if item.get('is_active') else '启用'}", "callback_data": f"adm:content:toggle_daily:{item.get('id')}:{toggle_flag}"}])
                buttons.extend(TGAdminMenuBuilder.content_section_menu("daily"))
                self._send_text(admin_chat_id, TGAdminFormatter.format_material_admin_list("🌤️ 日常素材管理", items, "daily"), buttons)
                return
            if section == "new_daily_material":
                self._activate_text_input(admin_chat_id, business_account_id, "content_new_daily_material", TGAdminFormatter.format_content_create_prompt("🌤️ 新建日常素材", ["分类 | 优先级（可选，默认100） | 场景标签逗号（可选） | 文本内容", "如果需要媒体链接，可在末尾继续追加 | 媒体URL"]), {"back_callback": "adm:content:daily"}, [[{"text": "🔙 返回日常素材", "callback_data": "adm:content:daily"}]])
                return
            if section == "toggle_daily" and len(parts) >= 5 and parts[3].isdigit():
                material_id = int(parts[3]); is_active = parts[4] == "1"
                result = self.admin_api_service.toggle_daily_material_active(material_id, is_active)
                message = "✅ 已更新日常素材状态。" if result.get("ok") else "❌ 更新日常素材状态失败。"
                self._send_text(admin_chat_id, message, [[{"text": "🔙 返回日常素材", "callback_data": "adm:content:daily"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
                return
            if section == "project_materials":
                projects = self.admin_api_service.list_projects_admin(business_account_id, include_inactive=False)
                lines = ["📦 项目素材管理", "", "选择项目查看或新增素材："]
                buttons: list[list[dict[str, str]]] = []
                for item in projects[:12]:
                    lines.append(f"- {item.get('id')}: {item.get('name') or '-'}")
                    buttons.append([{"text": f"项目 {item.get('id')} 素材", "callback_data": f"adm:content:project_materials_list:{item.get('id')}"}])
                buttons.extend(TGAdminMenuBuilder.content_section_menu("project_materials"))
                self._send_text(admin_chat_id, "\n".join(lines), buttons)
                return
            if section == "project_materials_list" and len(parts) >= 4 and parts[3].isdigit():
                project_id = int(parts[3])
                items = self.admin_api_service.list_project_materials_admin(project_id, include_inactive=True)
                buttons: list[list[dict[str, str]]] = []
                for item in items[:10]:
                    toggle_flag = "0" if item.get("is_active") else "1"
                    buttons.append([{"text": f"素材 {item.get('id')} {'停用' if item.get('is_active') else '启用'}", "callback_data": f"adm:content:toggle_project_material:{item.get('id')}:{toggle_flag}:project:{project_id}"}])
                buttons.append([{"text": "➕ 新建项目素材", "callback_data": f"adm:content:new_project_material:{project_id}"}])
                buttons.append([{"text": "🔙 返回项目素材管理", "callback_data": "adm:content:project_materials"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}])
                self._send_text(admin_chat_id, TGAdminFormatter.format_material_admin_list(f"📦 项目素材管理 | 项目 {project_id}", items, "project"), buttons)
                return
            if section == "new_project_material":
                project_id = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else None
                prompt_lines = ["项目ID | 类型 | 优先级（可选，默认100） | 场景标签逗号（可选） | 文本内容", "如果需要媒体链接，可在末尾继续追加 | 媒体URL"]
                prompt_title = "📦 新建项目素材"
                payload = {"back_callback": f"adm:content:project_materials_list:{project_id}" if project_id else "adm:content:project_materials"}
                if project_id:
                    payload["project_id"] = project_id
                    prompt_lines[0] = f"{project_id} | 类型 | 优先级（可选，默认100） | 场景标签逗号（可选） | 文本内容"
                self._activate_text_input(admin_chat_id, business_account_id, "content_new_project_material", TGAdminFormatter.format_content_create_prompt(prompt_title, prompt_lines), payload, [[{"text": "🔙 返回项目素材管理", "callback_data": payload['back_callback']} ]])
                return
            if section == "toggle_project_material" and len(parts) >= 5 and parts[3].isdigit():
                material_id = int(parts[3]); is_active = parts[4] == "1"
                project_id = int(parts[6]) if len(parts) >= 7 and parts[5] == "project" and parts[6].isdigit() else None
                result = self.admin_api_service.toggle_project_material_active(material_id, is_active)
                message = "✅ 已更新项目素材状态。" if result.get("ok") else "❌ 更新项目素材状态失败。"
                back = f"adm:content:project_materials_list:{project_id}" if project_id else "adm:content:project_materials"
                self._send_text(admin_chat_id, message, [[{"text": "🔙 返回项目素材", "callback_data": back}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
                return
            receipt_buttons = []
            if user_id is not None:
                receipt_buttons.append([{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}])
            receipt_buttons.append([{"text": "🏠 主菜单", "callback_data": "adm:main"}])
            self._send_text(admin_chat_id, message, receipt_buttons)
            return
        self._send_text(admin_chat_id, f"⚠️ 暂不支持该按钮：{callback_data}")


class TGAdminHandlers:
    def __init__(self, tg_admin_callback_router: TGAdminCallbackRouter, tg_sender, admin_session_repo: AdminSessionRepository,
                 business_account_repo: BusinessAccountRepository, admin_api_service: AdminAPIService) -> None:
        self.tg_admin_callback_router = tg_admin_callback_router
        self.tg_sender = tg_sender
        self.admin_session_repo = admin_session_repo
        self.business_account_repo = business_account_repo
        self.admin_api_service = admin_api_service

    def _send_text(self, admin_chat_id: int, text: str, reply_markup=None) -> None:
        if reply_markup:
            self.tg_sender.send_admin_message(admin_chat_id, text, reply_markup)
        else:
            self.tg_sender.send_admin_text(admin_chat_id, text)

    @staticmethod
    def _parse_multi_value_input(message_text: str) -> list[str]:
        normalized = (message_text or "").replace("\n", ",").replace(" ", ",")
        result: list[str] = []
        for item in normalized.split(","):
            item = item.strip()
            if item and item not in result:
                result.append(item)
        return result

    def _record_action(self, business_account_id: int | None, admin_chat_id: int, action_type: str, message_text: str,
                       target_user_id: int | None = None, target_project_id: int | None = None,
                       payload: dict[str, Any] | None = None, result_status: str = 'success') -> None:
        self.admin_api_service.record_admin_action(
            business_account_id=business_account_id,
            admin_chat_id=admin_chat_id,
            action_type=action_type,
            message_text=message_text,
            target_user_id=target_user_id,
            target_project_id=target_project_id,
            payload=payload,
            result_status=result_status,
        )

    def _search_customers(self, business_account_id: int, query: str, limit: int = 10) -> list[dict]:
        normalized = (query or "").strip()
        if not normalized:
            return []
        pattern = f"%{normalized}%"
        db = self.admin_api_service.conversation_repo.db
        with db.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT u.id AS user_id, u.tg_user_id, u.display_name, u.username
                FROM users u
                JOIN conversations c ON c.user_id=u.id AND c.business_account_id=%s
                WHERE u.tg_user_id=%s OR CAST(u.id AS TEXT)=%s OR COALESCE(u.username,'') ILIKE %s OR COALESCE(u.display_name,'') ILIKE %s
                ORDER BY u.id DESC
                LIMIT %s
                """,
                (business_account_id, normalized, normalized, pattern, pattern, limit),
            )
            return cur.fetchall()

    def handle_admin_command(self, admin_chat_id: int, command_text: str, operator: str = "admin") -> None:
        text_value = (command_text or "").strip().lower()
        if text_value in ("/admin", "/startadmin", "admin"):
            self.admin_session_repo.clear_session(admin_chat_id)
            menu = TGAdminMenuBuilder.main_menu()
            self._send_text(admin_chat_id, menu["text"], menu["reply_markup"])
            return
        self._send_text(admin_chat_id, "⚠️ 未知的管理员命令，请使用 /admin")

    def handle_admin_callback(self, admin_chat_id: int, callback_data: str, operator: str = "admin") -> None:
        self.tg_admin_callback_router.handle(admin_chat_id, callback_data, operator)

    def handle_admin_text(self, admin_chat_id: int, text: str, operator: str = "admin") -> None:
        message_text = (text or "").strip()
        if not message_text:
            self._send_text(admin_chat_id, "⚠️ 请输入有效内容。")
            return
        if message_text.lower() in ("/admin", "/startadmin", "admin"):
            self.handle_admin_command(admin_chat_id, message_text, operator)
            return
        session = self.admin_session_repo.get_session(admin_chat_id)
        if not session:
            self._send_text(admin_chat_id, "ℹ️ 当前没有等待中的文本操作。请先使用 /admin 打开菜单。")
            return
        state = (session.get("state") or "idle").strip()
        payload = session.get("payload_json") or {}
        business_account_id = int(payload.get("business_account_id") or session.get("business_account_id") or 0)
        back_callback = str(payload.get("back_callback") or "adm:main")
        if state == "search_customer":
            rows = self._search_customers(business_account_id, message_text, 10)
            if not rows:
                self._send_text(admin_chat_id, "🔎 未找到匹配客户。")
            else:
                lines = ["🔎 搜索结果", ""]
                buttons: list[list[dict[str, str]]] = []
                for row in rows[:10]:
                    lines.append(f"用户ID: {row.get('user_id')} | TG: {row.get('tg_user_id')} | 昵称: {row.get('display_name') or '-'} | 用户名: {row.get('username') or '-'}")
                    buttons.append([{"text": f"查看用户 {row.get('user_id')}", "callback_data": f"adm:customer:{row.get('user_id')}:adm:main"}])
                buttons.append([{"text": "🏠 主菜单", "callback_data": "adm:main"}])
                self._send_text(admin_chat_id, "\n".join(lines), buttons)
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "manual_takeover_target":
            if not message_text.isdigit():
                self._send_text(admin_chat_id, "⚠️ 请输入数字形式的会话ID。")
                return
            conversation_id = int(message_text)
            detail = self.admin_api_service.get_customer_detail_by_conversation(conversation_id)
            if not detail.get("ok"):
                self._send_text(admin_chat_id, "❌ 未找到该会话。")
                self.admin_session_repo.clear_session(admin_chat_id)
                return
            user_id = int(((detail.get("profile") or {}).get("user") or {}).get("id"))
            result = self.admin_api_service.start_manual_handover(business_account_id, user_id, conversation_id, operator, reason="telegram admin manual takeover")
            message = "✅ 已开始人工接管。" if result.get("ok") else f"❌ 接管失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "manual_takeover", message, target_user_id=user_id, payload={"conversation_id": conversation_id}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message)
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "resume_ai_target":
            if not message_text.isdigit():
                self._send_text(admin_chat_id, "⚠️ 请输入数字形式的会话ID。")
                return
            conversation_id = int(message_text)
            result = self.admin_api_service.resume_ai(conversation_id, operator)
            message = "✅ 已恢复 AI 自动聊天。" if result.get("ok") else f"❌ 恢复失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "resume_ai", message, payload={"conversation_id": conversation_id}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message)
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "set_project_target":
            user_id = int(payload.get("user_id") or 0)
            projects = self.admin_api_service.list_projects_for_business_account(business_account_id)
            normalized = message_text.lower()
            project_id: int | None
            if normalized in ("0", "none", "clear", "null"):
                project_id = None
            else:
                if not message_text.isdigit():
                    self._send_text(admin_chat_id, "⚠️ 请输入项目ID，或输入 0 / none / clear 清空项目。")
                    return
                project_id = int(message_text)
                valid_ids = {int(item.get('id')) for item in projects}
                if project_id not in valid_ids:
                    self._send_text(admin_chat_id, "⚠️ 项目ID无效，请重新输入。")
                    return
            result = self.admin_api_service.set_project(business_account_id, user_id, project_id, operator, reason_text="telegram admin manual project update")
            project_name = next((item.get('name') for item in projects if int(item.get('id')) == project_id), None) if project_id else None
            message = f"✅ 已更新项目为 {project_name or '空'}。" if result.get("ok") else f"❌ 更新项目失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "set_project", message, target_user_id=user_id, target_project_id=project_id, payload={"project_id": project_id}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "add_tag_target":
            user_id = int(payload.get("user_id") or 0)
            available_tags = set(self.admin_api_service.list_all_tag_names(business_account_id))
            requested_tags = self._parse_multi_value_input(message_text)
            valid_tags = [tag for tag in requested_tags if tag in available_tags]
            invalid_tags = [tag for tag in requested_tags if tag not in available_tags]
            added: list[str] = []
            for tag_name in valid_tags:
                result = self.admin_api_service.add_tag(business_account_id, user_id, tag_name, operator, reason_text="telegram admin manual tag add")
                if result.get("ok"):
                    added.append(tag_name)
            message_lines = ["🏷️ 标签添加结果", "", f"已添加: {', '.join(added) if added else '-'}"]
            if invalid_tags:
                message_lines.append(f"无效标签: {', '.join(invalid_tags)}")
            message = "\n".join(message_lines)
            self._record_action(business_account_id, admin_chat_id, "add_tag", message, target_user_id=user_id, payload={"added_tags": added, "invalid_tags": invalid_tags}, result_status='success' if added else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "remove_tag_target":
            user_id = int(payload.get("user_id") or 0)
            active_tags = set(self.admin_api_service.list_active_tag_names_for_user(business_account_id, user_id))
            requested_tags = self._parse_multi_value_input(message_text)
            removable_tags = [tag for tag in requested_tags if tag in active_tags]
            invalid_tags = [tag for tag in requested_tags if tag not in active_tags]
            removed: list[str] = []
            for tag_name in removable_tags:
                result = self.admin_api_service.remove_tag(business_account_id, user_id, tag_name)
                if result.get("ok"):
                    removed.append(tag_name)
            message_lines = ["🗑️ 标签移除结果", "", f"已移除: {', '.join(removed) if removed else '-'}"]
            if invalid_tags:
                message_lines.append(f"不在当前标签中的项: {', '.join(invalid_tags)}")
            message = "\n".join(message_lines)
            self._record_action(business_account_id, admin_chat_id, "remove_tag", message, target_user_id=user_id, payload={"removed_tags": removed, "invalid_tags": invalid_tags}, result_status='success' if removed else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "content_new_project":
            fields = self._split_pipe_fields(message_text)
            if len(fields) < 1 or not fields[0]:
                self._send_text(admin_chat_id, "⚠️ 请输入：名称 | 描述（可选）")
                return
            name = fields[0]
            description = fields[1] if len(fields) > 1 and fields[1] else None
            result = self.admin_api_service.create_project(business_account_id, name, description)
            message = f"✅ 已新建项目：{(result.get('project') or {}).get('name') or name}" if result.get("ok") else f"❌ 新建项目失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "content_new_project", message, target_project_id=((result.get('project') or {}).get('id') if result.get('ok') else None), payload={"name": name}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回项目列表", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "content_new_segment":
            fields = self._split_pipe_fields(message_text)
            fallback_project_id = int(payload.get("project_id") or 0) or None
            if fallback_project_id:
                if len(fields) < 2:
                    self._send_text(admin_chat_id, "⚠️ 请输入：阶段名称 | 描述（可选） | 排序（可选）")
                    return
                project_id = fallback_project_id
                name = fields[0]
                description = fields[1] if len(fields) > 1 and fields[1] else None
                sort_order = int(fields[2]) if len(fields) > 2 and fields[2].strip().isdigit() else 0
            else:
                if len(fields) < 2 or not fields[0].isdigit():
                    self._send_text(admin_chat_id, "⚠️ 请输入：项目ID | 阶段名称 | 描述（可选） | 排序（可选）")
                    return
                project_id = int(fields[0])
                name = fields[1]
                description = fields[2] if len(fields) > 2 and fields[2] else None
                sort_order = int(fields[3]) if len(fields) > 3 and fields[3].strip().isdigit() else 0
            result = self.admin_api_service.create_project_segment(project_id, name, description, sort_order)
            message = f"✅ 已新建阶段：{(result.get('segment') or {}).get('name') or name}" if result.get("ok") else f"❌ 新建阶段失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "content_new_segment", message, target_project_id=project_id, payload={"project_id": project_id, "name": name, "sort_order": sort_order}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回项目阶段", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "content_new_marketing":
            fields = self._split_pipe_fields(message_text)
            if len(fields) < 3 or not fields[0].isdigit():
                self._send_text(admin_chat_id, "⚠️ 请输入：项目ID | 分类 | 优先级（可选） | 内容")
                return
            project_id = int(fields[0])
            category = fields[1] or "general"
            if len(fields) >= 4 and fields[2].strip().isdigit():
                priority = int(fields[2].strip())
                content_text = " | ".join(fields[3:]).strip()
            else:
                priority = 100
                content_text = " | ".join(fields[2:]).strip()
            if not content_text:
                self._send_text(admin_chat_id, "⚠️ 内容不能为空。")
                return
            result = self.admin_api_service.create_marketing_entry(project_id, category, content_text, priority)
            message = f"✅ 已新增营销方案 / 文案，ID：{(result.get('entry') or {}).get('id') or '-'}" if result.get("ok") else f"❌ 新建营销方案失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "content_new_marketing", message, target_project_id=project_id, payload={"project_id": project_id, "category": category, "priority": priority}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回营销方案库", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "content_new_persona_material":
            fields = self._split_pipe_fields(message_text)
            if len(fields) < 4:
                self._send_text(admin_chat_id, "⚠️ 请输入：类型 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）")
                return
            material_type = fields[0] or "text"
            if fields[1].strip().isdigit():
                priority = int(fields[1].strip())
                tag_text = fields[2]
                content_text = fields[3]
                media_url = fields[4] if len(fields) > 4 and fields[4] else None
            else:
                priority = 100
                tag_text = fields[1]
                content_text = fields[2]
                media_url = fields[3] if len(fields) > 3 and fields[3] else None
            result = self.admin_api_service.create_persona_material(business_account_id, material_type, content_text, media_url, self._parse_scene_tags(tag_text), priority)
            message = f"✅ 已新增人设素材，ID：{(result.get('material') or {}).get('id') or '-'}" if result.get("ok") else f"❌ 新增人设素材失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "content_new_persona_material", message, payload={"material_type": material_type, "priority": priority}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回人设素材", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "content_new_daily_material":
            fields = self._split_pipe_fields(message_text)
            if len(fields) < 4:
                self._send_text(admin_chat_id, "⚠️ 请输入：分类 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）")
                return
            category = fields[0] or "general"
            if fields[1].strip().isdigit():
                priority = int(fields[1].strip())
                tag_text = fields[2]
                content_text = fields[3]
                media_url = fields[4] if len(fields) > 4 and fields[4] else None
            else:
                priority = 100
                tag_text = fields[1]
                content_text = fields[2]
                media_url = fields[3] if len(fields) > 3 and fields[3] else None
            result = self.admin_api_service.create_daily_material(business_account_id, category, content_text, media_url, self._parse_scene_tags(tag_text), priority)
            message = f"✅ 已新增日常素材，ID：{(result.get('material') or {}).get('id') or '-'}" if result.get("ok") else f"❌ 新增日常素材失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "content_new_daily_material", message, payload={"category": category, "priority": priority}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回日常素材", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "content_new_project_material":
            fields = self._split_pipe_fields(message_text)
            fallback_project_id = int(payload.get("project_id") or 0) or None
            if fallback_project_id:
                if len(fields) < 4:
                    self._send_text(admin_chat_id, "⚠️ 请输入：类型 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）")
                    return
                project_id = fallback_project_id
                material_type = fields[0] or "text"
                if fields[1].strip().isdigit():
                    priority = int(fields[1].strip())
                    tag_text = fields[2]
                    content_text = fields[3]
                    media_url = fields[4] if len(fields) > 4 and fields[4] else None
                else:
                    priority = 100
                    tag_text = fields[1]
                    content_text = fields[2]
                    media_url = fields[3] if len(fields) > 3 and fields[3] else None
            else:
                if len(fields) < 5 or not fields[0].isdigit():
                    self._send_text(admin_chat_id, "⚠️ 请输入：项目ID | 类型 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）")
                    return
                project_id = int(fields[0])
                material_type = fields[1] or "text"
                if fields[2].strip().isdigit():
                    priority = int(fields[2].strip())
                    tag_text = fields[3]
                    content_text = fields[4]
                    media_url = fields[5] if len(fields) > 5 and fields[5] else None
                else:
                    priority = 100
                    tag_text = fields[2]
                    content_text = fields[3]
                    media_url = fields[4] if len(fields) > 4 and fields[4] else None
            result = self.admin_api_service.create_project_material(project_id, material_type, content_text, media_url, self._parse_scene_tags(tag_text), priority)
            message = f"✅ 已新增项目素材，ID：{(result.get('material') or {}).get('id') or '-'}" if result.get("ok") else f"❌ 新增项目素材失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "content_new_project_material", message, target_project_id=project_id, payload={"project_id": project_id, "material_type": material_type, "priority": priority}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回项目素材", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        if state == "note_input":
            self._send_text(admin_chat_id, f"📝 备注已记录：{message_text[:200]}")
            self.admin_session_repo.clear_session(admin_chat_id)
            return
        self.admin_session_repo.clear_session(admin_chat_id)
        self._send_text(admin_chat_id, "ℹ️ 当前状态已失效，请重新打开 /admin 菜单。")

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

    @bp.get("/content/projects/<int:business_account_id>")
    def content_projects(business_account_id: int):
        return jsonify({"ok": True, "data": admin_api_service.list_projects_admin(business_account_id, include_inactive=True)})

    @bp.get("/content/marketing/<int:business_account_id>")
    def content_marketing(business_account_id: int):
        project_id = request.args.get("project_id", default=None, type=int)
        return jsonify({"ok": True, "data": admin_api_service.list_marketing_entries(business_account_id, project_id, include_inactive=True)})

    @bp.get("/content/persona-materials/<int:business_account_id>")
    def content_persona_materials(business_account_id: int):
        return jsonify({"ok": True, "data": admin_api_service.list_persona_materials_admin(business_account_id, include_inactive=True)})

    @bp.get("/content/daily-materials/<int:business_account_id>")
    def content_daily_materials(business_account_id: int):
        return jsonify({"ok": True, "data": admin_api_service.list_daily_materials_admin(business_account_id, include_inactive=True)})

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
    outbound_sender_worker = app_components.get("outbound_sender_worker")
    if outbound_sender_worker is not None:
        outbound_sender_worker.start()

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
            return jsonify({"ok": True, "dispatched": dispatched})
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
    event_lock_repo = WebhookEventLockRepository(db)
    admin_session_repo = AdminSessionRepository(db)
    outbound_job_repo = OutboundJobRepository(db)
    business_message_edit_repo = BusinessMessageEditRepository(db)
    runtime_policy_repo = RuntimePolicyRepository(db, settings.default_timezone)
    pacing_state_repo = PacingStateRepository(db)
    proactive_followup_repo = ProactiveFollowupPlanRepository(db)
    admin_action_receipt_repo = AdminActionReceiptRepository(db)

    openai_adapter = OpenAIClientAdapter(settings.openai_api_key)
    llm_service = LLMService(openai_adapter, settings.llm_model_name)
    sender_service = SenderService(outbound_job_repo, immediate_sender=TelegramBusinessSenderAdapter(tg_client))
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
    work_hours_engine = WorkHoursEngine(settings.default_timezone)
    proactive_followup_engine = ProactiveFollowupEngine()

    ai_switch_engine = AISwitchEngine(settings_repo, user_control_repo, runtime_policy_repo, work_hours_engine)
    project_classifier = ProjectClassifier(project_repo)
    project_segment_manager = ProjectSegmentManager(project_repo)
    tagging_engine = TaggingEngine()
    intent_engine = IntentEngine()
    human_escalation_engine = HumanEscalationEngine()
    ops_category_manager = OpsCategoryManager()

    content_selector = ContentSelector(material_repo, ScriptSelector(script_repo), MaterialSelector(material_repo), PersonaMaterialSelector(), DailyMaterialSelector(), MaterialPackageSelector(material_repo))

    handover_manager = HandoverManager(handover_repo, user_control_repo, conversation_repo, admin_queue_repo)
    handover_summary_builder = HandoverSummaryBuilder(handover_repo, conversation_repo, llm_service)
    resume_chat_manager = ResumeChatManager(handover_repo, conversation_repo, user_control_repo)
    memory_manager = ConversationMemoryManager(conversation_repo, llm_service)

    customer_actions = CustomerActions(user_control_repo, user_repo, handover_manager, handover_summary_builder, resume_chat_manager, handover_repo, conversation_repo)
    admin_api_service = AdminAPIService(user_repo, receipt_repo, handover_repo, conversation_repo, user_control_repo, admin_queue_repo, customer_actions, resume_chat_manager, project_repo, script_repo, admin_action_receipt_repo, material_repo)
    dashboard_service = DashboardService(settings_repo, admin_queue_repo, receipt_repo, handover_repo, conversation_repo)
    tg_admin_callback_router = TGAdminCallbackRouter(admin_api_service, dashboard_service, tg_client, business_account_repo, admin_session_repo)
    tg_admin_handlers = TGAdminHandlers(tg_admin_callback_router, tg_client, admin_session_repo, business_account_repo, admin_api_service)

    orchestrator = Orchestrator(
        conversation_repo, user_repo, settings_repo, user_control_repo, business_account_repo,
        bootstrap_repo, receipt_repo, admin_queue_repo, material_repo, project_repo, script_repo,
        persona_core, persona_profile_builder, understanding_engine, stage_engine,
        mode_router, reply_planner, reply_style_engine, reply_self_check_engine,
        reply_delay_engine, ai_switch_engine, project_classifier, project_segment_manager,
        tagging_engine, intent_engine, human_escalation_engine, ops_category_manager,
        content_selector, sender_service, memory_manager,
        runtime_policy_repo, pacing_state_repo, proactive_followup_repo,
        work_hours_engine, proactive_followup_engine,
        handover_repo, admin_notifier,
    )

    gateway = TelegramBusinessGateway(settings, orchestrator.handle_inbound_message, business_account_repo, event_lock_repo, business_message_edit_repo)
    outbound_sender_worker = OutboundSenderWorker(settings)
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
        "outbound_sender_worker": outbound_sender_worker,
    }


def main() -> None:
    settings = Settings.load()
    settings.validate()
    setup_logging(settings.log_level)
    logger.info("Starting application.")
    app_components = build_app_components(settings)

    outbound_sender_worker = app_components.get("outbound_sender_worker")
    if outbound_sender_worker is not None:
        outbound_sender_worker.start()

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


# step10 note: final entrypoint moved to file end so late-stage extensions are applied before startup.

# =========================
# step8 extensions
# =========================

_previous_initialize_database = initialize_database
_previous_build_material_delivery_plan = build_material_delivery_plan
_BaseUserRepository = UserRepository
_BaseMaterialRepository = MaterialRepository
_BaseAdminAPIService = AdminAPIService
_BaseTGAdminFormatter = TGAdminFormatter
_BaseTGAdminMenuBuilder = TGAdminMenuBuilder
_BaseTGAdminCallbackRouter = TGAdminCallbackRouter
_BaseTGAdminHandlers = TGAdminHandlers
_BaseContentSelector = ContentSelector
_BaseTelegramBotAPIClient = TelegramBotAPIClient
_BaseOutboundSenderWorker = OutboundSenderWorker
_BaseOrchestrator = Orchestrator

LANGUAGE_NAME_BY_CODE = {"en": "English", "de": "German", "fr": "French"}
LANGUAGE_CODE_BY_NAME = {v.lower(): k for k, v in LANGUAGE_NAME_BY_CODE.items()}


def normalize_language_code(value: str | None) -> str | None:
    raw = str(value or "").strip().lower().replace("_", "-")
    if not raw:
        return None
    for prefix in ("en", "de", "fr"):
        if raw == prefix or raw.startswith(prefix + "-"):
            return prefix
    return None


def language_code_to_name(value: str | None) -> str:
    return LANGUAGE_NAME_BY_CODE.get(normalize_language_code(value) or "en", "English")


def detect_language_snapshot(text: str | None, telegram_language_code: str | None = None) -> dict[str, Any]:
    normalized_text = str(text or "").strip()
    telegram_code = normalize_language_code(telegram_language_code)
    if not normalized_text:
        return {
            "language_code": telegram_code or "en",
            "confidence": 0.45 if telegram_code else 0.2,
            "source": "telegram_profile" if telegram_code else "fallback",
        }
    lowered = normalized_text.lower()
    scores = {"en": 0.0, "de": 0.0, "fr": 0.0}
    english_tokens = {"the", "and", "you", "your", "with", "for", "that", "have", "just", "okay", "thanks", "hello", "today", "tomorrow"}
    german_tokens = {"und", "ich", "nicht", "ist", "danke", "bitte", "heute", "morgen", "hallo", "gut", "aber", "oder", "schon", "noch", "gerne", "möchte"}
    french_tokens = {"bonjour", "merci", "avec", "pour", "vous", "aujourd", "demain", "salut", "être", "avoir", "pas", "bien", "très", "peux", "peut"}
    token_list = [token.strip(".,!?;:-_()[]{}\"'") for token in lowered.split() if token.strip()]
    for token in token_list:
        if token in english_tokens:
            scores["en"] += 1.0
        if token in german_tokens:
            scores["de"] += 1.0
        if token in french_tokens:
            scores["fr"] += 1.0
    if any(ch in lowered for ch in "äöüß"):
        scores["de"] += 2.5
    if any(ch in lowered for ch in "àâçéèêëîïôùûüÿœæ"):
        scores["fr"] += 2.5
    if telegram_code:
        scores[telegram_code] += 0.75
    best_code = max(scores, key=scores.get)
    best_score = scores[best_code]
    total = sum(scores.values()) or 1.0
    confidence = max(0.35, min(0.99, best_score / total if total else 0.35)) if best_score > 0 else (0.55 if telegram_code else 0.35)
    source = "telegram_profile" if best_score <= 0 and telegram_code else "heuristic"
    if best_score <= 0 and telegram_code:
        best_code = telegram_code
    return {"language_code": best_code, "confidence": round(confidence, 3), "source": source}


def resolve_user_reply_language(user_profile: dict[str, Any]) -> str:
    user_row = (user_profile or {}).get("user") or {}
    mode = (user_row.get("reply_language_mode") or "fixed_en").lower()
    if mode == "fixed_en":
        return "English"
    if mode == "auto_follow":
        candidate = normalize_language_code(user_row.get("detected_language")) or normalize_language_code(user_row.get("preferred_language")) or normalize_language_code(user_row.get("language_code"))
        return language_code_to_name(candidate)
    preferred = normalize_language_code(user_row.get("preferred_language")) or "en"
    return language_code_to_name(preferred)


def initialize_database(db: Database) -> None:
    _previous_initialize_database(db)
    extra_statements = [
        '''
        CREATE TABLE IF NOT EXISTS voice_templates (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NOT NULL REFERENCES business_accounts(id) ON DELETE CASCADE,
            category TEXT NOT NULL DEFAULT 'general',
            title TEXT NOT NULL,
            media_url TEXT NOT NULL,
            script_text TEXT NULL,
            language_code VARCHAR(10) NOT NULL DEFAULT 'en',
            scene_tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            priority INT NOT NULL DEFAULT 100,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_voice_templates_business_active
            ON voice_templates (business_account_id, is_active, priority, id)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_voice_templates_language
            ON voice_templates (business_account_id, language_code, is_active)
        ''',
    ]
    with db.transaction():
        with db.cursor() as cur:
            for stmt in extra_statements:
                cur.execute(stmt)


class UserRepository(_BaseUserRepository):
    def set_reply_language_mode(self, user_id: int, mode: str, preferred_language: str | None = None, source: str = "admin_set") -> None:
        mode_value = (mode or "fixed_en").strip().lower()
        preferred = (preferred_language or "").strip().lower() or None
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET reply_language_mode=%s,
                        preferred_language=COALESCE(%s, preferred_language),
                        language_source=%s,
                        updated_at=NOW()
                    WHERE id=%s
                    """,
                    (mode_value, preferred, source, user_id),
                )

    def update_detected_language(self, user_id: int, detected_language: str | None, confidence: float | None,
                                 source: str = "auto_detected") -> None:
        normalized_language = normalize_language_code(detected_language)
        should_apply = normalized_language is not None
        applied_source = source if should_apply else None
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET detected_language=%s,
                        language_confidence=%s,
                        language_source=CASE
                            WHEN reply_language_mode='auto_follow' AND %s::boolean THEN %s::text
                            ELSE language_source
                        END,
                        preferred_language=CASE
                            WHEN reply_language_mode='auto_follow' AND %s::boolean THEN %s::text
                            ELSE preferred_language
                        END,
                        updated_at=NOW()
                    WHERE id=%s
                    """,
                    (normalized_language, confidence, should_apply, applied_source, should_apply, normalized_language, user_id),
                )


class MaterialRepository(_BaseMaterialRepository):
    def get_voice_templates(self, business_account_id: int, language_code: str | None = None, include_inactive: bool = False) -> list[dict]:
        conditions = ["business_account_id=%s"]
        params: list[Any] = [business_account_id]
        if not include_inactive:
            conditions.append("is_active=TRUE")
        normalized_language = normalize_language_code(language_code)
        if normalized_language:
            conditions.append("(language_code=%s OR language_code='' OR language_code='multi')")
            params.append(normalized_language)
        query = f"SELECT * FROM voice_templates WHERE {' AND '.join(conditions)} ORDER BY priority ASC, id ASC"
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

    def list_voice_templates_admin(self, business_account_id: int, include_inactive: bool = True) -> list[dict]:
        query = "SELECT * FROM voice_templates WHERE business_account_id=%s"
        params: list[Any] = [business_account_id]
        if not include_inactive:
            query += " AND is_active=TRUE"
        query += " ORDER BY priority ASC, id ASC"
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            return [self._normalize_scene_tags(row) for row in cur.fetchall()]

    def create_voice_template(self, business_account_id: int, category: str, title: str, media_url: str,
                              language_code: str = 'en', script_text: str | None = None,
                              scene_tags: list[str] | None = None, priority: int = 100,
                              is_active: bool = True) -> dict:
        normalized_language = normalize_language_code(language_code) or 'en'
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO voice_templates (
                        business_account_id, category, title, media_url, script_text,
                        language_code, scene_tags_json, priority, is_active, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,NOW(),NOW()) RETURNING *
                    """,
                    (business_account_id, category, title, media_url, script_text, normalized_language,
                     json.dumps(scene_tags or [], ensure_ascii=False), priority, is_active),
                )
                return self._normalize_scene_tags(cur.fetchone())

    def update_voice_template_active(self, template_id: int, is_active: bool) -> bool:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    "UPDATE voice_templates SET is_active=%s, updated_at=NOW() WHERE id=%s",
                    (is_active, template_id),
                )
                return cur.rowcount > 0

    def get_material_library_overview(self, business_account_id: int, project_id: int | None = None) -> dict[str, Any]:
        overview = super().get_material_library_overview(business_account_id, project_id)
        overview["voice_template_count"] = len(self.get_voice_templates(business_account_id))
        return overview


class VoiceTemplateSelector:
    def select(self, templates: list[dict[str, Any]], mode: str, reply_language: str, limit: int = 1) -> list[dict[str, Any]]:
        desired_code = normalize_language_code(LANGUAGE_CODE_BY_NAME.get(str(reply_language or '').lower())) or normalize_language_code(reply_language) or 'en'
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in templates or []:
            language_code = normalize_language_code(item.get('language_code'))
            score = 0.0
            if language_code == desired_code:
                score += 3.0
            elif not language_code or language_code == 'multi':
                score += 1.0
            category = str(item.get('category') or '').lower()
            title = str(item.get('title') or '').lower()
            if mode and (mode in category or mode in title):
                score += 1.5
            tags = [str(v).lower() for v in (item.get('scene_tags_json') or [])]
            if mode and mode.lower() in tags:
                score += 1.0
            score += max(0.0, 1.0 - min(float(item.get('priority') or 100), 200.0) / 200.0)
            scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1].get('priority') or 100), int(pair[1].get('id') or 0)))
        return [item for _, item in scored[:max(0, limit)]]


class ContentSelector(_BaseContentSelector):
    def __init__(self, material_repo: MaterialRepository, script_selector: ScriptSelector, material_selector: MaterialSelector,
                 persona_material_selector: PersonaMaterialSelector, daily_material_selector: DailyMaterialSelector,
                 material_package_selector: MaterialPackageSelector, voice_template_selector: VoiceTemplateSelector | None = None) -> None:
        super().__init__(material_repo, script_selector, material_selector, persona_material_selector, daily_material_selector, material_package_selector)
        self.voice_template_selector = voice_template_selector or VoiceTemplateSelector()

    def _build_prompt_summary(self, selected: dict[str, Any]) -> str:
        base = super()._build_prompt_summary(selected)
        voice_count = len(selected.get("voice_templates") or [])
        if voice_count:
            return f"{base} ; Voice templates: {voice_count}"[:600]
        return base

    def select(self, business_account_id: int, project_id: int | None, mode: str, reply_plan: ReplyPlan) -> dict:
        selected = super().select(business_account_id, project_id, mode, reply_plan)
        selected["voice_templates"] = self.voice_template_selector.select(
            self.material_repo.get_voice_templates(business_account_id),
            mode,
            "English",
            limit=1 if reply_plan.should_send_material else 0,
        )
        overview = selected.get("material_library_overview") or {}
        overview["voice_template_count"] = len(self.material_repo.get_voice_templates(business_account_id))
        selected["material_library_overview"] = overview
        selected["prompt_summary"] = self._build_prompt_summary(selected)
        return selected


def _delivery_steps_from_voice_template(item: dict[str, Any]) -> list[dict[str, Any]]:
    media_url = str(item.get("media_url") or "").strip()
    if not media_url:
        return []
    origin = {
        "scope": "voice_template",
        "voice_template_id": item.get("id"),
        "title": item.get("title"),
        "language_code": item.get("language_code"),
    }
    steps: list[dict[str, Any]] = [{
        "step_type": "voice",
        "media_url": media_url,
        "caption": str(item.get("script_text") or item.get("title") or "").strip() or None,
        "origin": origin,
    }]
    extra_text = str(item.get("script_text") or "").strip()
    if extra_text and len(extra_text) > 140:
        steps.append({
            "step_type": "text",
            "text": extra_text,
            "buttons": [],
            "origin": origin,
        })
    return steps


def build_material_delivery_plan(selected_content: dict[str, Any], should_send_material: bool) -> dict[str, Any]:
    plan = _previous_build_material_delivery_plan(selected_content, should_send_material)
    if not should_send_material:
        return plan
    extra_steps = list(plan.get("steps") or [])
    for item in (selected_content.get("voice_templates") or [])[:1]:
        extra_steps.extend(_delivery_steps_from_voice_template(item))
    plan["steps"] = extra_steps[:10]
    plan["summary"] = _summarize_delivery_plan({"steps": plan["steps"]})
    return plan


class TelegramBotAPIClient(_BaseTelegramBotAPIClient):
    def send_voice(self, conversation_id: int, media_url: str, caption: str | None = None, reply_markup=None) -> None:
        payload = {
            "chat_id": conversation_id,
            "voice": media_url,
            "caption": caption,
            "reply_markup": self._prepare_reply_markup(reply_markup),
        }
        self._call_api("sendVoice", payload)


class OutboundSenderWorker(_BaseOutboundSenderWorker):
    def _execute_delivery_plan(self, tg_client: TelegramBotAPIClient, conversation_id: int, delivery_plan: dict[str, Any] | None) -> None:
        plan = delivery_plan or {}
        for step in plan.get("steps") or []:
            step_type = str(step.get("step_type") or "").strip().lower()
            if step_type == "voice":
                buttons = tg_client.build_inline_keyboard(step.get("buttons") or [])
                tg_client.send_voice(conversation_id, str(step.get("media_url") or ""), caption=step.get("caption"), reply_markup=buttons)
                time.sleep(0.35)
                continue
            if step_type == "text":
                text_value = str(step.get("text") or "").strip()
                buttons = tg_client.build_inline_keyboard(step.get("buttons") or [])
                if text_value or buttons:
                    tg_client.send_text(conversation_id, text_value or "More details here.", reply_markup=buttons)
            elif step_type == "photo":
                buttons = tg_client.build_inline_keyboard(step.get("buttons") or [])
                tg_client.send_photo(conversation_id, str(step.get("media_url") or ""), caption=step.get("caption"), reply_markup=buttons)
            elif step_type == "video":
                buttons = tg_client.build_inline_keyboard(step.get("buttons") or [])
                tg_client.send_video(conversation_id, str(step.get("media_url") or ""), caption=step.get("caption"), reply_markup=buttons)
            elif step_type == "media_group":
                media_items = list(step.get("media") or [])
                if media_items:
                    tg_client.send_media_group(conversation_id, media_items)
            else:
                logger.warning("Unsupported delivery step_type=%s", step_type)
            time.sleep(0.35)


class AdminAPIService(_BaseAdminAPIService):
    def list_voice_templates_admin(self, business_account_id: int, include_inactive: bool = True) -> list[dict]:
        return [] if not self.material_repo else self.material_repo.list_voice_templates_admin(business_account_id, include_inactive)

    def create_voice_template(self, business_account_id: int, category: str, title: str, media_url: str,
                              language_code: str = 'en', script_text: str | None = None,
                              scene_tags: list[str] | None = None, priority: int = 100) -> dict:
        if not self.material_repo:
            return {"ok": False, "reason": "material repository unavailable"}
        row = self.material_repo.create_voice_template(business_account_id, category, title, media_url, language_code, script_text, scene_tags, priority)
        return {"ok": True, "voice_template": row}

    def toggle_voice_template_active(self, template_id: int, is_active: bool) -> dict:
        if not self.material_repo:
            return {"ok": False, "reason": "material repository unavailable"}
        return {"ok": bool(self.material_repo.update_voice_template_active(template_id, is_active))}

    def set_user_language_mode(self, business_account_id: int, user_id: int, mode: str,
                               preferred_language: str | None, operator: str) -> dict:
        detail = self.get_customer_detail(business_account_id, user_id)
        if not detail.get("ok"):
            return {"ok": False, "reason": "user not found"}
        self.user_repo.set_reply_language_mode(user_id, mode, preferred_language, source='admin_set')
        return {"ok": True, "profile": self.user_repo.get_user_profile(business_account_id, user_id)}


class TGAdminFormatter(_BaseTGAdminFormatter):
    @staticmethod
    def format_customer_detail(detail: dict) -> str:
        base = _BaseTGAdminFormatter.format_customer_detail(detail)
        profile = (detail.get("profile") or {})
        user_row = profile.get("user") or {}
        language_line = f"语言模式: {user_row.get('reply_language_mode') or '-'} | 偏好: {user_row.get('preferred_language') or '-'} | 检测: {user_row.get('detected_language') or '-'} | 置信度: {user_row.get('language_confidence') if user_row.get('language_confidence') is not None else '-'}"
        if "运营分类:" in base and language_line not in base:
            base = base.replace("运营分类:", language_line + "\n运营分类:", 1)
        material_overview = detail.get("material_library_overview") or {}
        if material_overview and "语音模板" not in base:
            needle = f"素材库: 人设{material_overview.get('persona_material_count', 0)} | 日常{material_overview.get('daily_material_count', 0)} | 项目{material_overview.get('project_material_count', 0)}"
            replacement = needle + f" | 语音模板{material_overview.get('voice_template_count', 0)}"
            base = base.replace(needle, replacement, 1)
        return base

    @staticmethod
    def format_voice_template_admin_list(items: list[dict]) -> str:
        if not items:
            return "🎙️ 语音模板库\n\n暂无内容。"
        lines = ["🎙️ 语音模板库", "", "当前模板："]
        for item in items[:20]:
            status = "启用" if item.get("is_active") else "停用"
            lines.append(f"- {item.get('id')}: {item.get('title') or '-'} | 分类 {item.get('category') or '-'} | 语言 {item.get('language_code') or '-'} | 优先级 {item.get('priority', '-')} | {status}")
            if item.get('script_text'):
                lines.append(f"  {(item.get('script_text') or '-')[:88]}")
            lines.append(f"  语音: {(item.get('media_url') or '-')[:88]}")
        return "\n".join(lines)


class TGAdminMenuBuilder(_BaseTGAdminMenuBuilder):
    @staticmethod
    def content_menu() -> dict:
        base = _BaseTGAdminMenuBuilder.content_menu()
        reply_markup = list(base.get("reply_markup") or [])
        reply_markup.insert(3, [{"text": "🎙️ 语音模板", "callback_data": "adm:content:voice_templates"}])
        base["reply_markup"] = reply_markup
        return base

    @staticmethod
    def content_section_menu(section: str) -> list[list[dict[str, str]]]:
        if section == "voice_templates":
            return [
                [{"text": "➕ 新建语音模板", "callback_data": "adm:content:new_voice_template"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ]
        return _BaseTGAdminMenuBuilder.content_section_menu(section)

    @staticmethod
    def customer_action_menu(user_id: int, conversation_id: int | None, back_callback: str = "adm:main") -> list[list[dict[str, str]]]:
        rows = _BaseTGAdminMenuBuilder.customer_action_menu(user_id, conversation_id, back_callback)
        rows.insert(-1, [
            {"text": "🌐 自动跟随", "callback_data": f"adm:lang:{user_id}:auto_follow:auto:{back_callback}"},
            {"text": "🇬🇧 固定英文", "callback_data": f"adm:lang:{user_id}:manual_fixed:en:{back_callback}"},
        ])
        rows.insert(-1, [
            {"text": "🇩🇪 固定德语", "callback_data": f"adm:lang:{user_id}:manual_fixed:de:{back_callback}"},
            {"text": "🇫🇷 固定法语", "callback_data": f"adm:lang:{user_id}:manual_fixed:fr:{back_callback}"},
        ])
        return rows


class TGAdminCallbackRouter(_BaseTGAdminCallbackRouter):
    def handle(self, admin_chat_id: int, callback_data: str, operator: str = "admin") -> None:
        parts = (callback_data or "").split(":")
        business_account_id = self._resolve_business_account_id()
        if parts[:2] == ["adm", "lang"] and len(parts) >= 5 and business_account_id is not None:
            user_id = int(parts[2])
            mode = parts[3]
            preferred_language = None if parts[4] == 'auto' else parts[4]
            back_callback = self._tail(parts, 5)
            result = self.admin_api_service.set_user_language_mode(business_account_id, user_id, mode, preferred_language, operator)
            lang_label = preferred_language or 'auto'
            message = f"✅ 已更新语言模式为 {mode} / {lang_label}。" if result.get('ok') else f"❌ 更新语言模式失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "set_user_language_mode", message, target_user_id=user_id, payload={"mode": mode, "preferred_language": preferred_language}, result_status='success' if result.get('ok') else 'failed')
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回客户详情", "callback_data": f"adm:customer:{user_id}:{back_callback}"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            return
        if parts[:3] == ["adm", "content", "voice_templates"] and business_account_id is not None:
            items = self.admin_api_service.list_voice_templates_admin(business_account_id, include_inactive=True)
            buttons: list[list[dict[str, str]]] = []
            for item in items[:10]:
                toggle_flag = "0" if item.get("is_active") else "1"
                buttons.append([{"text": f"模板 {item.get('id')} {'停用' if item.get('is_active') else '启用'}", "callback_data": f"adm:content:toggle_voice_template:{item.get('id')}:{toggle_flag}"}])
            buttons.extend(TGAdminMenuBuilder.content_section_menu("voice_templates"))
            self._send_text(admin_chat_id, TGAdminFormatter.format_voice_template_admin_list(items), buttons)
            return
        if parts[:3] == ["adm", "content", "new_voice_template"] and business_account_id is not None:
            self._activate_text_input(admin_chat_id, business_account_id, "content_new_voice_template", TGAdminFormatter.format_content_create_prompt("🎙️ 新建语音模板", ["分类 | 语言(en/de/fr) | 优先级（可选，默认100） | 标题 | 语音URL | 说明文本（可选）", "可选继续追加 | 场景标签逗号"]), {"back_callback": "adm:content:voice_templates"}, [[{"text": "🔙 返回语音模板", "callback_data": "adm:content:voice_templates"}]])
            return
        if len(parts) >= 5 and parts[:3] == ["adm", "content", "toggle_voice_template"] and parts[3].isdigit() and business_account_id is not None:
            template_id = int(parts[3])
            is_active = parts[4] == "1"
            result = self.admin_api_service.toggle_voice_template_active(template_id, is_active)
            message = "✅ 已更新语音模板状态。" if result.get("ok") else "❌ 更新语音模板状态失败。"
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回语音模板", "callback_data": "adm:content:voice_templates"}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            return
        return super().handle(admin_chat_id, callback_data, operator)


class TGAdminHandlers(_BaseTGAdminHandlers):
    def handle_admin_text(self, admin_chat_id: int, text_value: str, operator: str = "admin") -> None:
        session = self.admin_session_repo.get_session(admin_chat_id) or {}
        if session.get("state") == "content_new_voice_template":
            payload = session.get("payload_json") or {}
            business_account_id = int(payload.get("business_account_id") or session.get("business_account_id") or self.business_account_repo.get_default_account_id() or 0)
            fields = [part.strip() for part in (text_value or '').split('|')]
            try:
                if len(fields) < 5:
                    raise ValueError('format')
                category = fields[0] or 'general'
                language_code = normalize_language_code(fields[1]) or 'en'
                priority = int(fields[2] or '100') if len(fields) > 2 and fields[2] else 100
                title = fields[3]
                media_url = fields[4]
                script_text = fields[5] if len(fields) > 5 and fields[5] else None
                scene_tags = self._parse_multi_value_input(fields[6] if len(fields) > 6 else '')
            except Exception:
                self._send_text(admin_chat_id, "❌ 格式错误，请按“分类 | 语言 | 优先级 | 标题 | 语音URL | 说明文本(可选) | 场景标签(可选)”重新输入。")
                return
            result = self.admin_api_service.create_voice_template(business_account_id, category, title, media_url, language_code, script_text, scene_tags, priority)
            message = f"✅ 已创建语音模板 {((result.get('voice_template') or {}).get('id') or '')}。" if result.get('ok') else f"❌ 创建语音模板失败：{result.get('reason') or '未知错误'}"
            self._record_action(business_account_id, admin_chat_id, "content_new_voice_template", message, payload={"category": category, "language_code": language_code, "priority": priority}, result_status='success' if result.get('ok') else 'failed')
            self.admin_session_repo.clear_session(admin_chat_id)
            back_callback = payload.get('back_callback') or 'adm:content:voice_templates'
            self._send_text(admin_chat_id, message, [[{"text": "🔙 返回语音模板", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
            return
        return super().handle_admin_text(admin_chat_id, text_value, operator)


class Orchestrator(_BaseOrchestrator):
    def handle_inbound_message(self, inbound_message: InboundMessage) -> None:
        try:
            self.business_account_repo.create_if_not_exists(
                inbound_message.tg_business_account_id,
                f"BusinessAccount-{inbound_message.tg_business_account_id}",
                None,
            )
            user_payload = inbound_message.raw_payload.get("business_message") or inbound_message.raw_payload.get("edited_business_message") or inbound_message.raw_payload.get("message") or {}
            from_user = user_payload.get("from") or {}
            user_id = self.user_repo.get_or_create_user(
                inbound_message.tg_user_id,
                display_name=(from_user.get("first_name") or f"User-{inbound_message.tg_user_id}"),
                username=from_user.get("username"),
                language_code=from_user.get("language_code"),
            )
            detection = detect_language_snapshot(inbound_message.text or "", from_user.get("language_code"))
            self.user_repo.update_detected_language(user_id, detection.get("language_code"), detection.get("confidence"), detection.get("source") or 'auto_detected')
        except Exception:
            logger.exception("Step8 language detection pre-processing failed.")
        return super().handle_inbound_message(inbound_message)


# =========================
# step9: content management polish + brain polish
# =========================

_Step9BaseProjectRepository = ProjectRepository
_Step9BaseScriptRepository = ScriptRepository
_Step9BaseMaterialRepository = MaterialRepository
_Step9BaseAdminAPIService = AdminAPIService
_Step9BaseTGAdminFormatter = TGAdminFormatter
_Step9BaseTGAdminMenuBuilder = TGAdminMenuBuilder
_Step9BaseTGAdminCallbackRouter = TGAdminCallbackRouter
_Step9BaseTGAdminHandlers = TGAdminHandlers
_Step9BaseConversationMemoryManager = ConversationMemoryManager
_Step9BaseProactiveFollowupEngine = ProactiveFollowupEngine
_Step9BaseReplyPlanner = ReplyPlanner
_Step9BaseReplyStyleEngine = ReplyStyleEngine


def _step9_soft_clip(text_value: str, limit: int = 260) -> str:
    text_value = (text_value or '').strip()
    if len(text_value) <= limit:
        return text_value
    clipped = text_value[:limit].rsplit(' ', 1)[0].strip()
    return clipped + '…' if clipped else text_value[:limit] + '…'


def _step9_compact_list(items: Any, limit: int = 4, item_limit: int = 50) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        value = _step9_soft_clip(str(item), item_limit)
        lowered = value.lower()
        if not value or lowered in seen:
            continue
        seen.add(lowered)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _step9_boundary_signal_from_text(text_value: str) -> str | None:
    normalized = f" {(text_value or '').lower()} "
    stop_keys = [' stop ', ' don\'t text ', ' do not text ', ' leave me alone ', ' 不要再发 ', ' 别发了 ', ' 不要联系 ', '别联系', '别打扰', ' no more ', ' not interested ']
    sleep_keys = [' sleep ', ' sleeping ', ' good night ', ' going to bed ', ' 晚安 ', ' 睡了 ', ' 睡觉 ', ' rest now ']
    busy_keys = [' busy ', ' in a meeting ', ' at work ', ' working ', ' occupied ', ' 在忙 ', ' 工作中 ', ' 开会 ', ' 忙着 ', ' later pls ']
    later_keys = [' later ', ' talk later ', ' text later ', ' tomorrow ', ' 之后聊 ', ' 晚点 ', ' 稍后 ', ' 回头 ', ' later please ']
    if any(key in normalized for key in stop_keys):
        return 'stop'
    if any(key in normalized for key in sleep_keys):
        return 'sleep'
    if any(key in normalized for key in busy_keys):
        return 'busy'
    if any(key in normalized for key in later_keys):
        return 'later'
    return None


class ProjectRepository(_Step9BaseProjectRepository):
    def update_project_details(self, project_id: int, name: str, description: str | None = None) -> dict | None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE projects
                    SET name=%s, description=%s, updated_at=NOW()
                    WHERE id=%s
                    RETURNING *
                    """,
                    (name.strip(), description, project_id),
                )
                return cur.fetchone()

    def archive_project(self, project_id: int) -> bool:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE projects SET is_active=FALSE, updated_at=NOW() WHERE id=%s", (project_id,))
                return cur.rowcount > 0

    def update_project_segment_details(self, segment_id: int, name: str, description: str | None = None, sort_order: int | None = None) -> dict | None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE project_segments
                    SET name=%s,
                        description=%s,
                        sort_order=COALESCE(%s, sort_order),
                        updated_at=NOW()
                    WHERE id=%s
                    RETURNING *
                    """,
                    (name.strip(), description, sort_order, segment_id),
                )
                return cur.fetchone()

    def archive_project_segment(self, segment_id: int) -> bool:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE project_segments SET is_active=FALSE, updated_at=NOW() WHERE id=%s", (segment_id,))
                return cur.rowcount > 0


class ScriptRepository(_Step9BaseScriptRepository):
    def update_project_script(self, script_id: int, category: str, content_text: str, priority: int | None = None) -> dict | None:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE project_scripts
                    SET category=%s,
                        content_text=%s,
                        priority=COALESCE(%s, priority),
                        updated_at=NOW()
                    WHERE id=%s
                    RETURNING *
                    """,
                    (category.strip(), content_text, priority, script_id),
                )
                return cur.fetchone()

    def update_project_script_priority(self, script_id: int, priority: int) -> bool:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute("UPDATE project_scripts SET priority=%s, updated_at=NOW() WHERE id=%s", (priority, script_id))
                return cur.rowcount > 0

    def archive_project_script(self, script_id: int) -> bool:
        return self.update_project_script_active(script_id, False)


class MaterialRepository(_Step9BaseMaterialRepository):
    def _get_material_row(self, table_name: str, row_id: int) -> dict | None:
        if table_name not in {"persona_materials", "daily_materials", "project_materials", "voice_templates"}:
            return None
        with self.db.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE id=%s LIMIT 1", (row_id,))
            row = cur.fetchone()
        return self._normalize_scene_tags(row) if row else None

    def _update_material_row(self, table_name: str, row_id: int, columns: dict[str, Any]) -> dict | None:
        if table_name not in {"persona_materials", "daily_materials", "project_materials", "voice_templates"}:
            return None
        if not columns:
            return self._get_material_row(table_name, row_id)
        clauses: list[str] = []
        values: list[Any] = []
        for key, value in columns.items():
            if key == 'scene_tags_json':
                clauses.append(f"{key}=%s::jsonb")
                values.append(json.dumps(value or [], ensure_ascii=False))
            else:
                clauses.append(f"{key}=%s")
                values.append(value)
        values.append(row_id)
        query = f"UPDATE {table_name} SET {', '.join(clauses)}, updated_at=NOW() WHERE id=%s RETURNING *"
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(query, tuple(values))
                row = cur.fetchone()
        return self._normalize_scene_tags(row) if row else None

    def update_persona_material(self, material_id: int, material_type: str, content_text: str,
                                media_url: str | None = None, scene_tags: list[str] | None = None,
                                priority: int | None = None) -> dict | None:
        return self._update_material_row("persona_materials", material_id, {
            "material_type": material_type,
            "content_text": content_text,
            "media_url": media_url,
            "scene_tags_json": scene_tags or [],
            "priority": priority,
        })

    def update_daily_material(self, material_id: int, category: str, content_text: str,
                              media_url: str | None = None, scene_tags: list[str] | None = None,
                              priority: int | None = None) -> dict | None:
        return self._update_material_row("daily_materials", material_id, {
            "category": category,
            "content_text": content_text,
            "media_url": media_url,
            "scene_tags_json": scene_tags or [],
            "priority": priority,
        })

    def update_project_material(self, material_id: int, material_type: str, content_text: str,
                                media_url: str | None = None, scene_tags: list[str] | None = None,
                                priority: int | None = None) -> dict | None:
        return self._update_material_row("project_materials", material_id, {
            "material_type": material_type,
            "content_text": content_text,
            "media_url": media_url,
            "scene_tags_json": scene_tags or [],
            "priority": priority,
        })

    def update_voice_template(self, template_id: int, category: str, title: str, media_url: str,
                              language_code: str = 'en', script_text: str | None = None,
                              scene_tags: list[str] | None = None, priority: int | None = None) -> dict | None:
        return self._update_material_row("voice_templates", template_id, {
            "category": category,
            "title": title,
            "media_url": media_url,
            "language_code": normalize_language_code(language_code) or 'en',
            "script_text": script_text,
            "scene_tags_json": scene_tags or [],
            "priority": priority,
        })

    def update_material_priority(self, table_name: str, row_id: int, priority: int) -> bool:
        if table_name not in {"persona_materials", "daily_materials", "project_materials", "voice_templates"}:
            return False
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(f"UPDATE {table_name} SET priority=%s, updated_at=NOW() WHERE id=%s", (priority, row_id))
                return cur.rowcount > 0

    def archive_material(self, table_name: str, row_id: int) -> bool:
        if table_name == 'voice_templates':
            return self.update_voice_template_active(row_id, False)
        return self.update_material_active(table_name, row_id, False)


class AdminAPIService(_Step9BaseAdminAPIService):
    def update_project(self, project_id: int, name: str, description: str | None = None) -> dict:
        row = self.project_repo.update_project_details(project_id, name, description) if self.project_repo else None
        return {"ok": bool(row), "project": row, "reason": None if row else "project not found"}

    def archive_project(self, project_id: int) -> dict:
        ok = bool(self.project_repo and self.project_repo.archive_project(project_id))
        return {"ok": ok, "reason": None if ok else "project not found"}

    def update_project_segment(self, segment_id: int, name: str, description: str | None = None, sort_order: int | None = None) -> dict:
        row = self.project_repo.update_project_segment_details(segment_id, name, description, sort_order) if self.project_repo else None
        return {"ok": bool(row), "segment": row, "reason": None if row else "segment not found"}

    def archive_project_segment(self, segment_id: int) -> dict:
        ok = bool(self.project_repo and self.project_repo.archive_project_segment(segment_id))
        return {"ok": ok, "reason": None if ok else "segment not found"}

    def update_marketing_entry(self, entry_id: int, category: str, content_text: str, priority: int | None = None) -> dict:
        row = self.script_repo.update_project_script(entry_id, category, content_text, priority) if self.script_repo else None
        return {"ok": bool(row), "entry": row, "reason": None if row else "entry not found"}

    def update_marketing_entry_priority(self, entry_id: int, priority: int) -> dict:
        ok = bool(self.script_repo and self.script_repo.update_project_script_priority(entry_id, priority))
        return {"ok": ok, "reason": None if ok else "entry not found"}

    def archive_marketing_entry(self, entry_id: int) -> dict:
        ok = bool(self.script_repo and self.script_repo.archive_project_script(entry_id))
        return {"ok": ok, "reason": None if ok else "entry not found"}

    def update_persona_material_admin(self, material_id: int, material_type: str, content_text: str,
                                      media_url: str | None = None, scene_tags: list[str] | None = None,
                                      priority: int | None = None) -> dict:
        row = self.material_repo.update_persona_material(material_id, material_type, content_text, media_url, scene_tags, priority) if self.material_repo else None
        return {"ok": bool(row), "material": row, "reason": None if row else "material not found"}

    def update_daily_material_admin(self, material_id: int, category: str, content_text: str,
                                    media_url: str | None = None, scene_tags: list[str] | None = None,
                                    priority: int | None = None) -> dict:
        row = self.material_repo.update_daily_material(material_id, category, content_text, media_url, scene_tags, priority) if self.material_repo else None
        return {"ok": bool(row), "material": row, "reason": None if row else "material not found"}

    def update_project_material_admin(self, material_id: int, material_type: str, content_text: str,
                                      media_url: str | None = None, scene_tags: list[str] | None = None,
                                      priority: int | None = None) -> dict:
        row = self.material_repo.update_project_material(material_id, material_type, content_text, media_url, scene_tags, priority) if self.material_repo else None
        return {"ok": bool(row), "material": row, "reason": None if row else "material not found"}

    def update_voice_template_admin(self, template_id: int, category: str, title: str, media_url: str,
                                    language_code: str = 'en', script_text: str | None = None,
                                    scene_tags: list[str] | None = None, priority: int | None = None) -> dict:
        row = self.material_repo.update_voice_template(template_id, category, title, media_url, language_code, script_text, scene_tags, priority) if self.material_repo else None
        return {"ok": bool(row), "voice_template": row, "reason": None if row else "voice template not found"}

    def update_material_priority(self, kind: str, row_id: int, priority: int) -> dict:
        table_map = {
            'persona': 'persona_materials',
            'daily': 'daily_materials',
            'project': 'project_materials',
            'voice': 'voice_templates',
        }
        table_name = table_map.get(kind)
        ok = bool(table_name and self.material_repo and self.material_repo.update_material_priority(table_name, row_id, priority))
        return {"ok": ok, "reason": None if ok else "material not found"}

    def archive_material_admin(self, kind: str, row_id: int) -> dict:
        table_map = {
            'persona': 'persona_materials',
            'daily': 'daily_materials',
            'project': 'project_materials',
            'voice': 'voice_templates',
        }
        table_name = table_map.get(kind)
        ok = bool(table_name and self.material_repo and self.material_repo.archive_material(table_name, row_id))
        return {"ok": ok, "reason": None if ok else "material not found"}


class TGAdminFormatter(_Step9BaseTGAdminFormatter):
    @staticmethod
    def format_material_admin_list(title: str, items: list[dict], category_label: str) -> str:
        if not items:
            return f"{title}\n\n暂无数据。"
        lines = [title, "", "当前内容："]
        for item in items[:20]:
            status = "启用" if item.get("is_active") else "停用"
            type_label = item.get("material_type") or item.get("category") or item.get("language_code") or category_label
            summary = _step9_soft_clip(str(item.get("content_text") or item.get("script_text") or item.get("title") or "-"), 72)
            lines.append(f"- {item.get('id')}: {type_label} | 优先级 {item.get('priority', 100)} | {status} | {summary}")
        lines.append("\n可继续使用编辑 / 归档 / 调整优先级。")
        return "\n".join(lines)

    @staticmethod
    def format_marketing_entry_list(items: list[dict]) -> str:
        if not items:
            return "📝 营销方案库\n\n暂无内容。"
        lines = ["📝 营销方案库", "", "当前文案 / 方案："]
        for item in items[:20]:
            status = "启用" if item.get("is_active") else "停用"
            lines.append(f"- {item.get('id')}: 项目 {item.get('project_id')} / {item.get('project_name') or '-'} | 分类 {item.get('category') or '-'} | 优先级 {item.get('priority', 100)} | {status} | {_step9_soft_clip(str(item.get('content_text') or '-'), 72)}")
        lines.append("\n可继续使用编辑 / 归档 / 调整优先级。")
        return "\n".join(lines)


class TGAdminMenuBuilder(_Step9BaseTGAdminMenuBuilder):
    @staticmethod
    def content_section_menu(section: str) -> list[list[dict[str, str]]]:
        if section == "projects":
            return [
                [{"text": "➕ 新建项目", "callback_data": "adm:content:new_project"}, {"text": "🧭 新建阶段", "callback_data": "adm:content:segments"}],
                [{"text": "✏️ 编辑项目", "callback_data": "adm:content:edit_project"}, {"text": "✏️ 编辑阶段", "callback_data": "adm:content:edit_segment"}],
                [{"text": "🗃️ 归档项目", "callback_data": "adm:content:delete_project"}, {"text": "🗃️ 归档阶段", "callback_data": "adm:content:delete_segment"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ]
        if section == "marketing":
            return [
                [{"text": "➕ 新建营销方案", "callback_data": "adm:content:new_marketing"}],
                [{"text": "✏️ 编辑方案", "callback_data": "adm:content:edit_marketing"}, {"text": "🎚️ 调优先级", "callback_data": "adm:content:priority_marketing"}],
                [{"text": "🗃️ 归档方案", "callback_data": "adm:content:delete_marketing"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ]
        if section == "persona":
            return [
                [{"text": "➕ 新建人设素材", "callback_data": "adm:content:new_persona_material"}],
                [{"text": "✏️ 编辑素材", "callback_data": "adm:content:edit_persona_material"}, {"text": "🎚️ 调优先级", "callback_data": "adm:content:priority_persona_material"}],
                [{"text": "🗃️ 归档素材", "callback_data": "adm:content:delete_persona_material"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ]
        if section == "daily":
            return [
                [{"text": "➕ 新建日常素材", "callback_data": "adm:content:new_daily_material"}],
                [{"text": "✏️ 编辑素材", "callback_data": "adm:content:edit_daily_material"}, {"text": "🎚️ 调优先级", "callback_data": "adm:content:priority_daily_material"}],
                [{"text": "🗃️ 归档素材", "callback_data": "adm:content:delete_daily_material"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ]
        if section == "project_materials":
            return [
                [{"text": "➕ 新建项目素材", "callback_data": "adm:content:new_project_material"}],
                [{"text": "✏️ 编辑素材", "callback_data": "adm:content:edit_project_material"}, {"text": "🎚️ 调优先级", "callback_data": "adm:content:priority_project_material"}],
                [{"text": "🗃️ 归档素材", "callback_data": "adm:content:delete_project_material"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ]
        if section == "voice_templates":
            return [
                [{"text": "➕ 新建语音模板", "callback_data": "adm:content:new_voice_template"}],
                [{"text": "✏️ 编辑模板", "callback_data": "adm:content:edit_voice_template"}, {"text": "🎚️ 调优先级", "callback_data": "adm:content:priority_voice_template"}],
                [{"text": "🗃️ 归档模板", "callback_data": "adm:content:delete_voice_template"}],
                [{"text": "🧱 返回内容管理", "callback_data": "adm:content"}, {"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ]
        return _Step9BaseTGAdminMenuBuilder.content_section_menu(section)


class TGAdminCallbackRouter(_Step9BaseTGAdminCallbackRouter):
    def _activate_content_input(self, admin_chat_id: int, business_account_id: int, state: str, title: str, lines: list[str], back_callback: str) -> None:
        self._activate_text_input(
            admin_chat_id,
            business_account_id,
            state,
            TGAdminFormatter.format_content_create_prompt(title, lines),
            {"back_callback": back_callback},
            [[{"text": "🔙 返回", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]],
        )

    def handle(self, admin_chat_id: int, callback_data: str, operator: str = "admin") -> None:
        parts = (callback_data or "").split(":")
        business_account_id = self._resolve_business_account_id()
        if parts[:3] == ["adm", "content", "edit_project"] and business_account_id is not None:
            self._activate_content_input(admin_chat_id, business_account_id, "content_edit_project", "📁 编辑项目", ["项目ID | 新名称 | 新描述（可选）"], "adm:content:projects")
            return
        if parts[:3] == ["adm", "content", "delete_project"] and business_account_id is not None:
            self._activate_content_input(admin_chat_id, business_account_id, "content_delete_project", "🗃️ 归档项目", ["项目ID"], "adm:content:projects")
            return
        if parts[:3] == ["adm", "content", "edit_segment"] and business_account_id is not None:
            self._activate_content_input(admin_chat_id, business_account_id, "content_edit_segment", "🧭 编辑项目阶段", ["阶段ID | 新名称 | 新描述（可选） | 排序（可选）"], "adm:content:projects")
            return
        if parts[:3] == ["adm", "content", "delete_segment"] and business_account_id is not None:
            self._activate_content_input(admin_chat_id, business_account_id, "content_delete_segment", "🗃️ 归档项目阶段", ["阶段ID"], "adm:content:projects")
            return
        mapping = {
            ("adm", "content", "edit_marketing"): ("content_edit_marketing", "📝 编辑营销方案", ["文案ID | 分类 | 优先级（可选） | 内容"], "adm:content:marketing"),
            ("adm", "content", "priority_marketing"): ("content_priority_marketing", "🎚️ 调整营销方案优先级", ["文案ID | 新优先级"], "adm:content:marketing"),
            ("adm", "content", "delete_marketing"): ("content_delete_marketing", "🗃️ 归档营销方案", ["文案ID"], "adm:content:marketing"),
            ("adm", "content", "edit_persona_material"): ("content_edit_persona_material", "👩 编辑人设素材", ["素材ID | 类型 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）"], "adm:content:persona"),
            ("adm", "content", "priority_persona_material"): ("content_priority_persona_material", "🎚️ 调整人设素材优先级", ["素材ID | 新优先级"], "adm:content:persona"),
            ("adm", "content", "delete_persona_material"): ("content_delete_persona_material", "🗃️ 归档人设素材", ["素材ID"], "adm:content:persona"),
            ("adm", "content", "edit_daily_material"): ("content_edit_daily_material", "🌤️ 编辑日常素材", ["素材ID | 分类 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）"], "adm:content:daily"),
            ("adm", "content", "priority_daily_material"): ("content_priority_daily_material", "🎚️ 调整日常素材优先级", ["素材ID | 新优先级"], "adm:content:daily"),
            ("adm", "content", "delete_daily_material"): ("content_delete_daily_material", "🗃️ 归档日常素材", ["素材ID"], "adm:content:daily"),
            ("adm", "content", "edit_project_material"): ("content_edit_project_material", "📦 编辑项目素材", ["素材ID | 类型 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）"], "adm:content:project_materials"),
            ("adm", "content", "priority_project_material"): ("content_priority_project_material", "🎚️ 调整项目素材优先级", ["素材ID | 新优先级"], "adm:content:project_materials"),
            ("adm", "content", "delete_project_material"): ("content_delete_project_material", "🗃️ 归档项目素材", ["素材ID"], "adm:content:project_materials"),
            ("adm", "content", "edit_voice_template"): ("content_edit_voice_template", "🎙️ 编辑语音模板", ["模板ID | 分类 | 语言 | 优先级（可选） | 标题 | 语音URL | 说明文本（可选） | 场景标签（可选）"], "adm:content:voice_templates"),
            ("adm", "content", "priority_voice_template"): ("content_priority_voice_template", "🎚️ 调整语音模板优先级", ["模板ID | 新优先级"], "adm:content:voice_templates"),
            ("adm", "content", "delete_voice_template"): ("content_delete_voice_template", "🗃️ 归档语音模板", ["模板ID"], "adm:content:voice_templates"),
        }
        key = tuple(parts[:3])
        if key in mapping and business_account_id is not None:
            state, title, lines, back_callback = mapping[key]
            self._activate_content_input(admin_chat_id, business_account_id, state, title, lines, back_callback)
            return
        return super().handle(admin_chat_id, callback_data, operator)


class TGAdminHandlers(_Step9BaseTGAdminHandlers):
    def _finalize_content_action(self, admin_chat_id: int, business_account_id: int, action_type: str,
                                 message: str, payload: dict[str, Any], back_callback: str,
                                 target_project_id: int | None = None, target_user_id: int | None = None,
                                 ok: bool = True) -> None:
        self._record_action(
            business_account_id,
            admin_chat_id,
            action_type,
            message,
            target_user_id=target_user_id,
            target_project_id=target_project_id,
            payload=payload,
            result_status='success' if ok else 'failed',
        )
        self.admin_session_repo.clear_session(admin_chat_id)
        self._send_text(admin_chat_id, message, [[{"text": "🔙 返回", "callback_data": back_callback}], [{"text": "🏠 主菜单", "callback_data": "adm:main"}]])

    def _parse_priority_pair(self, text_value: str) -> tuple[int, int] | None:
        fields = [part.strip() for part in (text_value or '').split('|')]
        if len(fields) < 2 or not fields[0].isdigit() or not re.match(r'^-?\d+$', fields[1] or ''):
            return None
        return int(fields[0]), int(fields[1])

    def handle_admin_text(self, admin_chat_id: int, text_value: str, operator: str = "admin") -> None:
        session = self.admin_session_repo.get_session(admin_chat_id) or {}
        state = str(session.get('state') or '')
        payload = session.get('payload_json') or {}
        business_account_id = int(payload.get('business_account_id') or session.get('business_account_id') or self.business_account_repo.get_default_account_id() or 0)
        back_callback = str(payload.get('back_callback') or 'adm:content')
        raw = (text_value or '').strip()
        if state == 'content_edit_project':
            fields = [part.strip() for part in raw.split('|')]
            if len(fields) < 2 or not fields[0].isdigit():
                self._send_text(admin_chat_id, '⚠️ 请输入：项目ID | 新名称 | 新描述（可选）')
                return
            project_id = int(fields[0]); name = fields[1]; description = fields[2] if len(fields) > 2 and fields[2] else None
            result = self.admin_api_service.update_project(project_id, name, description)
            self._finalize_content_action(admin_chat_id, business_account_id, state, f"✅ 已更新项目。" if result.get('ok') else f"❌ 更新项目失败：{result.get('reason') or '未知错误'}", {"project_id": project_id, "name": name}, back_callback, target_project_id=project_id, ok=bool(result.get('ok')))
            return
        if state == 'content_delete_project':
            if not raw.isdigit():
                self._send_text(admin_chat_id, '⚠️ 请输入项目ID。')
                return
            project_id = int(raw)
            result = self.admin_api_service.archive_project(project_id)
            self._finalize_content_action(admin_chat_id, business_account_id, state, '✅ 已归档项目。' if result.get('ok') else f"❌ 归档项目失败：{result.get('reason') or '未知错误'}", {"project_id": project_id}, back_callback, target_project_id=project_id, ok=bool(result.get('ok')))
            return
        if state == 'content_edit_segment':
            fields = [part.strip() for part in raw.split('|')]
            if len(fields) < 2 or not fields[0].isdigit():
                self._send_text(admin_chat_id, '⚠️ 请输入：阶段ID | 新名称 | 新描述（可选） | 排序（可选）')
                return
            segment_id = int(fields[0]); name = fields[1]; description = fields[2] if len(fields) > 2 and fields[2] else None
            sort_order = int(fields[3]) if len(fields) > 3 and re.match(r'^-?\d+$', fields[3]) else None
            result = self.admin_api_service.update_project_segment(segment_id, name, description, sort_order)
            self._finalize_content_action(admin_chat_id, business_account_id, state, '✅ 已更新项目阶段。' if result.get('ok') else f"❌ 更新阶段失败：{result.get('reason') or '未知错误'}", {"segment_id": segment_id, "name": name, "sort_order": sort_order}, back_callback, ok=bool(result.get('ok')))
            return
        if state == 'content_delete_segment':
            if not raw.isdigit():
                self._send_text(admin_chat_id, '⚠️ 请输入阶段ID。')
                return
            segment_id = int(raw)
            result = self.admin_api_service.archive_project_segment(segment_id)
            self._finalize_content_action(admin_chat_id, business_account_id, state, '✅ 已归档项目阶段。' if result.get('ok') else f"❌ 归档阶段失败：{result.get('reason') or '未知错误'}", {"segment_id": segment_id}, back_callback, ok=bool(result.get('ok')))
            return
        if state == 'content_edit_marketing':
            fields = [part.strip() for part in raw.split('|')]
            if len(fields) < 3 or not fields[0].isdigit():
                self._send_text(admin_chat_id, '⚠️ 请输入：文案ID | 分类 | 优先级（可选） | 内容')
                return
            entry_id = int(fields[0]); category = fields[1] or 'general'
            if len(fields) >= 4 and re.match(r'^-?\d+$', fields[2] or ''):
                priority = int(fields[2]); content_text = ' | '.join(fields[3:]).strip()
            else:
                priority = None; content_text = ' | '.join(fields[2:]).strip()
            result = self.admin_api_service.update_marketing_entry(entry_id, category, content_text, priority)
            self._finalize_content_action(admin_chat_id, business_account_id, state, '✅ 已更新营销方案。' if result.get('ok') else f"❌ 更新营销方案失败：{result.get('reason') or '未知错误'}", {"entry_id": entry_id, "category": category, "priority": priority}, back_callback, ok=bool(result.get('ok')))
            return
        if state == 'content_priority_marketing':
            pair = self._parse_priority_pair(raw)
            if not pair:
                self._send_text(admin_chat_id, '⚠️ 请输入：文案ID | 新优先级')
                return
            entry_id, priority = pair
            result = self.admin_api_service.update_marketing_entry_priority(entry_id, priority)
            self._finalize_content_action(admin_chat_id, business_account_id, state, '✅ 已更新营销方案优先级。' if result.get('ok') else f"❌ 更新优先级失败：{result.get('reason') or '未知错误'}", {"entry_id": entry_id, "priority": priority}, back_callback, ok=bool(result.get('ok')))
            return
        if state == 'content_delete_marketing':
            if not raw.isdigit():
                self._send_text(admin_chat_id, '⚠️ 请输入文案ID。')
                return
            entry_id = int(raw)
            result = self.admin_api_service.archive_marketing_entry(entry_id)
            self._finalize_content_action(admin_chat_id, business_account_id, state, '✅ 已归档营销方案。' if result.get('ok') else f"❌ 归档营销方案失败：{result.get('reason') or '未知错误'}", {"entry_id": entry_id}, back_callback, ok=bool(result.get('ok')))
            return

        material_edit_map = {
            'content_edit_persona_material': ('persona', self.admin_api_service.update_persona_material_admin, '⚠️ 请输入：素材ID | 类型 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）'),
            'content_edit_daily_material': ('daily', self.admin_api_service.update_daily_material_admin, '⚠️ 请输入：素材ID | 分类 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）'),
            'content_edit_project_material': ('project', self.admin_api_service.update_project_material_admin, '⚠️ 请输入：素材ID | 类型 | 优先级（可选） | 场景标签（可选） | 文本内容 | 媒体URL（可选）'),
        }
        if state in material_edit_map:
            kind, handler, error_message = material_edit_map[state]
            fields = [part.strip() for part in raw.split('|')]
            if len(fields) < 4 or not fields[0].isdigit():
                self._send_text(admin_chat_id, error_message)
                return
            row_id = int(fields[0]); head = fields[1]
            if len(fields) >= 5 and re.match(r'^-?\d+$', fields[2] or ''):
                priority = int(fields[2]); tag_text = fields[3]; content_text = fields[4]; media_url = fields[5] if len(fields) > 5 and fields[5] else None
            else:
                priority = None; tag_text = fields[2]; content_text = fields[3]; media_url = fields[4] if len(fields) > 4 and fields[4] else None
            tags = self._parse_scene_tags(tag_text)
            result = handler(row_id, head, content_text, media_url, tags, priority)
            label = {'persona': '人设素材', 'daily': '日常素材', 'project': '项目素材'}[kind]
            self._finalize_content_action(admin_chat_id, business_account_id, state, f'✅ 已更新{label}。' if result.get('ok') else f"❌ 更新{label}失败：{result.get('reason') or '未知错误'}", {"id": row_id, "priority": priority}, back_callback, ok=bool(result.get('ok')))
            return

        priority_map = {
            'content_priority_persona_material': ('persona', '人设素材'),
            'content_priority_daily_material': ('daily', '日常素材'),
            'content_priority_project_material': ('project', '项目素材'),
            'content_priority_voice_template': ('voice', '语音模板'),
        }
        if state in priority_map:
            pair = self._parse_priority_pair(raw)
            if not pair:
                self._send_text(admin_chat_id, '⚠️ 请输入：ID | 新优先级')
                return
            row_id, priority = pair
            kind, label = priority_map[state]
            result = self.admin_api_service.update_material_priority(kind, row_id, priority)
            self._finalize_content_action(admin_chat_id, business_account_id, state, f'✅ 已更新{label}优先级。' if result.get('ok') else f"❌ 更新{label}优先级失败：{result.get('reason') or '未知错误'}", {"id": row_id, "priority": priority}, back_callback, ok=bool(result.get('ok')))
            return

        delete_map = {
            'content_delete_persona_material': ('persona', '人设素材'),
            'content_delete_daily_material': ('daily', '日常素材'),
            'content_delete_project_material': ('project', '项目素材'),
            'content_delete_voice_template': ('voice', '语音模板'),
        }
        if state in delete_map:
            if not raw.isdigit():
                self._send_text(admin_chat_id, '⚠️ 请输入要归档的ID。')
                return
            row_id = int(raw)
            kind, label = delete_map[state]
            result = self.admin_api_service.archive_material_admin(kind, row_id)
            self._finalize_content_action(admin_chat_id, business_account_id, state, f'✅ 已归档{label}。' if result.get('ok') else f"❌ 归档{label}失败：{result.get('reason') or '未知错误'}", {"id": row_id}, back_callback, ok=bool(result.get('ok')))
            return

        if state == 'content_edit_voice_template':
            fields = [part.strip() for part in raw.split('|')]
            if len(fields) < 5 or not fields[0].isdigit():
                self._send_text(admin_chat_id, '⚠️ 请输入：模板ID | 分类 | 语言 | 优先级（可选） | 标题 | 语音URL | 说明文本（可选） | 场景标签（可选）')
                return
            template_id = int(fields[0]); category = fields[1] or 'general'; language_code = normalize_language_code(fields[2]) or 'en'
            offset = 3
            priority = None
            if len(fields) > 3 and re.match(r'^-?\d+$', fields[3] or ''):
                priority = int(fields[3]); offset = 4
            if len(fields) < offset + 2:
                self._send_text(admin_chat_id, '⚠️ 模板ID / 标题 / 语音URL不能为空。')
                return
            title = fields[offset]
            media_url = fields[offset + 1]
            script_text = fields[offset + 2] if len(fields) > offset + 2 and fields[offset + 2] else None
            scene_tags = self._parse_multi_value_input(fields[offset + 3] if len(fields) > offset + 3 else '')
            result = self.admin_api_service.update_voice_template_admin(template_id, category, title, media_url, language_code, script_text, scene_tags, priority)
            self._finalize_content_action(admin_chat_id, business_account_id, state, '✅ 已更新语音模板。' if result.get('ok') else f"❌ 更新语音模板失败：{result.get('reason') or '未知错误'}", {"template_id": template_id, "language_code": language_code, "priority": priority}, back_callback, ok=bool(result.get('ok')))
            return
        return super().handle_admin_text(admin_chat_id, text_value, operator)


class ConversationMemoryManager(_Step9BaseConversationMemoryManager):
    @staticmethod
    def build_memory_summary_text(long_term_memory: dict[str, Any] | None, latest_handover_summary: dict[str, Any] | None = None) -> str:
        memory = long_term_memory or {}
        if not memory and not latest_handover_summary:
            return ""
        parts: list[str] = []
        summary_text = _step9_soft_clip(str(memory.get("summary_text") or "").replace('\n', ' ').strip(), 220)
        if summary_text:
            parts.append(f"RollingSummary={summary_text}")
        key_facts = _step9_compact_list(memory.get("key_facts") or [], 4, 48)
        if key_facts:
            parts.append(f"KeyFacts={'; '.join(key_facts)}")
        preferences = _step9_compact_list(memory.get("user_preferences") or [], 4, 40)
        if preferences:
            parts.append(f"Preferences={'; '.join(preferences)}")
        boundaries = _step9_compact_list(memory.get("boundaries") or [], 4, 40)
        if boundaries:
            parts.append(f"Boundaries={'; '.join(boundaries)}")
        project_signals = _step9_compact_list(memory.get("project_signals") or [], 4, 40)
        if project_signals:
            parts.append(f"ProjectSignals={'; '.join(project_signals)}")
        followup_strategy = _step9_soft_clip(str(memory.get("followup_strategy") or "").strip(), 80)
        if followup_strategy:
            parts.append(f"Followup={followup_strategy}")
        if latest_handover_summary:
            resume_suggestion = _step9_soft_clip(str(latest_handover_summary.get("resume_suggestion") or "").strip(), 90)
            human_strategy = _step9_soft_clip(str(latest_handover_summary.get("human_strategy_summary") or "").strip(), 90)
            if resume_suggestion:
                parts.append(f"ResumeHint={resume_suggestion}")
            if human_strategy:
                parts.append(f"HumanStrategy={human_strategy}")
        return " | ".join(parts[:6])


class ReplyPlanner(_Step9BaseReplyPlanner):
    def plan(self, understanding: UnderstandingResult, stage_decision: StageDecision, mode_decision: ModeDecision, user_state: UserStateSnapshot) -> ReplyPlan:
        plan = super().plan(understanding, stage_decision, mode_decision, user_state)
        boundary = str(understanding.boundary_signal or '').lower()
        resistance = str(understanding.resistance_signal or '').lower()
        if boundary in ('busy', 'later', 'sleep'):
            plan.should_continue_product = False
            plan.should_send_material = False
            plan.should_leave_space = True
            plan.should_self_share = False
            plan.goal = 'pause_respect'
        if boundary == 'stop':
            plan.should_continue_product = False
            plan.should_send_material = False
            plan.should_self_share = False
            plan.should_leave_space = True
            plan.goal = 'pause_respect'
            plan.reason = (plan.reason or '') + ' | boundary=stop'
        if resistance in ('medium', 'high') and not understanding.explicit_product_query:
            plan.should_continue_product = False
            plan.should_send_material = False
            plan.should_leave_space = True
            plan.reason = (plan.reason or '') + f' | resistance={resistance}'
        return plan


class ReplyStyleEngine(_Step9BaseReplyStyleEngine):
    @staticmethod
    def _sanitize_output(text_value: str, boundary_signal: str | None, resistance_signal: str | None) -> str:
        text_value = (text_value or '').strip()
        if not text_value:
            return text_value
        boundary = str(boundary_signal or '').lower()
        resistance = str(resistance_signal or '').lower()
        if boundary == 'stop':
            if len(text_value.split()) > 24:
                text_value = '. '.join([seg.strip() for seg in re.split(r'[.!?]+', text_value) if seg.strip()][:1]).strip()
            text_value = _step9_soft_clip(text_value, 140)
        elif boundary in ('busy', 'later', 'sleep'):
            text_value = _step9_soft_clip(text_value, 200)
            text_value = re.sub(r'\?\s*\?+', '?', text_value)
            if text_value.count('?') > 1:
                first_q = text_value.find('?')
                text_value = text_value[:first_q + 1] + text_value[first_q + 1:].replace('?', '')
        elif resistance in ('medium', 'high'):
            text_value = text_value.replace('Let me know if you want to start right now.', 'We can keep it light and simple.')
            text_value = text_value.replace('Would you like me to walk you through the next step?', 'Happy to explain more whenever it feels right.')
            text_value = _step9_soft_clip(text_value, 240)
        return text_value.strip()

    def generate(self, latest_user_message: str, recent_context: list[dict], persona_summary: str, user_state_summary: str,
                 stage: str, chat_mode: str, understanding: dict, reply_plan: dict, selected_content: dict,
                 memory_summary: dict[str, Any] | None = None, reply_language: str = "English") -> str:
        text_value = super().generate(latest_user_message, recent_context, persona_summary, user_state_summary,
                                      stage, chat_mode, understanding, reply_plan, selected_content,
                                      memory_summary=memory_summary, reply_language=reply_language)
        return self._sanitize_output(text_value, (understanding or {}).get('boundary_signal'), (understanding or {}).get('resistance_signal'))


class ProactiveFollowupEngine(_Step9BaseProactiveFollowupEngine):
    def decide(self, policy: dict[str, Any], understanding: UnderstandingResult, mode_decision: ModeDecision,
               reply_plan: ReplyPlan, user_state: UserStateSnapshot, intent_decision: IntentDecision,
               context: ConversationContext, latest_user_text: str, project_name: str | None = None,
               latest_handover_summary: dict[str, Any] | None = None,
               escalation_decision: dict[str, Any] | None = None) -> FollowupDecision:
        boundary_override = _step9_boundary_signal_from_text(latest_user_text)
        if boundary_override == 'stop':
            now = utc_now()
            stop_snooze_hours = max(12, int((policy or {}).get('stop_snooze_hours') or 72))
            return FollowupDecision(False, reason='explicit stop boundary', pacing_mode='silent', snooze_until=now + timedelta(hours=stop_snooze_hours))
        if boundary_override in ('busy', 'later', 'sleep') and not understanding.boundary_signal:
            understanding = replace(understanding, boundary_signal=boundary_override)
        decision = super().decide(policy, understanding, mode_decision, reply_plan, user_state, intent_decision, context, latest_user_text, project_name, latest_handover_summary, escalation_decision)
        if not decision.should_schedule:
            return decision
        normalized_text = (latest_user_text or '').strip()
        if len(normalized_text) <= 3 and not understanding.high_intent_signal and not understanding.explicit_product_query:
            delayed = decision.scheduled_for + timedelta(hours=12) if decision.scheduled_for else None
            return replace(decision, scheduled_for=delayed, reason=(decision.reason or '') + ' | short_reply_extra_space', pacing_mode='gentle')
        if str(understanding.resistance_signal or '').lower() == 'high' and not understanding.high_intent_signal:
            return FollowupDecision(False, reason='high resistance suppress followup', pacing_mode='silent')
        return decision

# =========================
# step10 extensions
# =========================

_Step10BaseInitializeDatabase = initialize_database
_Step10BaseBuildAppComponents = build_app_components
_Step10BaseBuildAdminBlueprint = build_admin_blueprint
_Step10BaseTGAdminMenuBuilder = TGAdminMenuBuilder
_Step10BaseTGAdminCallbackRouter = TGAdminCallbackRouter
_Step10BaseMain = main
_STEP10_PREFLIGHT_SERVICE = None


def initialize_database(db: Database) -> None:
    _Step10BaseInitializeDatabase(db)
    extra_statements = [
        '''
        CREATE TABLE IF NOT EXISTS deployment_preflight_reports (
            id BIGSERIAL PRIMARY KEY,
            business_account_id BIGINT NULL REFERENCES business_accounts(id) ON DELETE SET NULL,
            trigger_source VARCHAR(32) NOT NULL DEFAULT 'manual',
            overall_status VARCHAR(16) NOT NULL,
            readiness_score INT NOT NULL DEFAULT 0,
            summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            checks_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_deployment_preflight_reports_account_created
            ON deployment_preflight_reports (business_account_id, created_at DESC)
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_deployment_preflight_reports_status_created
            ON deployment_preflight_reports (overall_status, created_at DESC)
        ''',
    ]
    with db.transaction():
        with db.cursor() as cur:
            for stmt in extra_statements:
                cur.execute(stmt)


class DeploymentPreflightRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create_report(self, business_account_id: int | None, trigger_source: str, overall_status: str,
                      readiness_score: int, summary: dict[str, Any], checks: list[dict[str, Any]]) -> int:
        with self.db.transaction():
            with self.db.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO deployment_preflight_reports (
                        business_account_id, trigger_source, overall_status,
                        readiness_score, summary_json, checks_json, created_at
                    ) VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,NOW())
                    RETURNING id
                    ''',
                    (
                        business_account_id,
                        (trigger_source or 'manual')[:32],
                        (overall_status or 'unknown')[:16],
                        int(readiness_score or 0),
                        json.dumps(summary or {}, ensure_ascii=False),
                        json.dumps(checks or [], ensure_ascii=False),
                    ),
                )
                row = cur.fetchone() or {}
        return int(row.get('id') or 0)

    def get_latest_report(self, business_account_id: int | None = None) -> dict[str, Any] | None:
        params: list[Any] = []
        query = 'SELECT * FROM deployment_preflight_reports'
        if business_account_id is not None:
            query += ' WHERE business_account_id=%s'
            params.append(business_account_id)
        query += ' ORDER BY created_at DESC, id DESC LIMIT 1'
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            row = cur.fetchone()
        if not row:
            return None
        data = dict(row)
        for key in ('summary_json', 'checks_json'):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value)
                except Exception:
                    data[key] = {} if key == 'summary_json' else []
        return data

    def list_recent_reports(self, business_account_id: int | None = None, limit: int = 10) -> list[dict[str, Any]]:
        params: list[Any] = []
        query = 'SELECT * FROM deployment_preflight_reports'
        if business_account_id is not None:
            query += ' WHERE business_account_id=%s'
            params.append(business_account_id)
        query += ' ORDER BY created_at DESC, id DESC LIMIT %s'
        params.append(max(1, limit))
        with self.db.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall() or []
        result: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            for key in ('summary_json', 'checks_json'):
                value = data.get(key)
                if isinstance(value, str):
                    try:
                        data[key] = json.loads(value)
                    except Exception:
                        data[key] = {} if key == 'summary_json' else []
            result.append(data)
        return result


class DeploymentPreflightService:
    REQUIRED_TABLES = [
        'business_accounts', 'users', 'conversations', 'messages', 'receipts', 'admin_queue',
        'webhook_event_locks', 'admin_sessions', 'outbound_jobs', 'admin_action_receipts',
        'business_message_edits', 'business_runtime_policies', 'conversation_pacing_state',
        'proactive_followup_plans', 'projects', 'project_segments', 'project_scripts',
        'project_materials', 'persona_materials', 'daily_materials', 'material_packages',
        'material_package_items', 'material_package_buttons', 'voice_templates',
        'deployment_preflight_reports',
    ]
    REQUIRED_USER_COLUMNS = [
        'reply_language_mode', 'preferred_language', 'detected_language', 'language_confidence', 'language_source'
    ]

    def __init__(self, settings: Settings, db: Database, business_account_repo: BusinessAccountRepository,
                 runtime_policy_repo: RuntimePolicyRepository, preflight_repo: DeploymentPreflightRepository) -> None:
        self.settings = settings
        self.db = db
        self.business_account_repo = business_account_repo
        self.runtime_policy_repo = runtime_policy_repo
        self.preflight_repo = preflight_repo

    def _scalar(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        with self.db.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone() or {}
        if isinstance(row, dict):
            return next(iter(row.values()), None)
        return row[0] if row else None

    def _fetchone(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
        with self.db.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone() or {}
        return dict(row or {})

    def _table_exists(self, table_name: str) -> bool:
        return bool(self._scalar('SELECT to_regclass(%s) IS NOT NULL AS ok', (table_name,)))

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        return bool(self._scalar(
            '''
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name=%s AND column_name=%s
            ) AS ok
            ''',
            (table_name, column_name),
        ))

    @staticmethod
    def _append(checks: list[dict[str, Any]], code: str, status: str, message: str,
                details: dict[str, Any] | None = None) -> None:
        checks.append({
            'code': code,
            'status': status,
            'message': message,
            'details': details or {},
        })

    @staticmethod
    def _compute_overall_status(checks: list[dict[str, Any]]) -> str:
        statuses = [str(item.get('status') or '').lower() for item in checks]
        if any(status == 'fail' for status in statuses):
            return 'fail'
        if any(status == 'warn' for status in statuses):
            return 'warn'
        return 'pass'

    @staticmethod
    def _compute_readiness_score(checks: list[dict[str, Any]]) -> int:
        if not checks:
            return 0
        total = 0.0
        for item in checks:
            status = str(item.get('status') or '').lower()
            if status == 'pass':
                total += 1.0
            elif status == 'warn':
                total += 0.55
        return max(0, min(100, int(round((total / len(checks)) * 100))))

    @staticmethod
    def _parse_jsonish(value: Any, fallback: Any) -> Any:
        if value is None:
            return fallback
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return fallback
        return fallback

    def _content_counts(self, business_account_id: int) -> dict[str, int]:
        counts = {
            'projects_active': int(self._scalar('SELECT COUNT(*) FROM projects WHERE business_account_id=%s AND is_active=TRUE', (business_account_id,)) or 0),
            'segments_active': int(self._scalar('SELECT COUNT(*) FROM project_segments WHERE business_account_id=%s AND is_active=TRUE', (business_account_id,)) or 0),
            'marketing_active': int(self._scalar('SELECT COUNT(*) FROM project_scripts WHERE business_account_id=%s AND is_active=TRUE', (business_account_id,)) or 0),
            'persona_materials_active': int(self._scalar('SELECT COUNT(*) FROM persona_materials WHERE business_account_id=%s AND is_active=TRUE', (business_account_id,)) or 0),
            'daily_materials_active': int(self._scalar('SELECT COUNT(*) FROM daily_materials WHERE business_account_id=%s AND is_active=TRUE', (business_account_id,)) or 0),
            'project_materials_active': int(self._scalar(
                '''
                SELECT COUNT(*) FROM project_materials pm
                JOIN projects p ON p.id=pm.project_id
                WHERE p.business_account_id=%s AND pm.is_active=TRUE
                ''',
                (business_account_id,),
            ) or 0),
            'material_packages_active': int(self._scalar('SELECT COUNT(*) FROM material_packages WHERE business_account_id=%s AND is_active=TRUE', (business_account_id,)) or 0),
            'voice_templates_active': int(self._scalar('SELECT COUNT(*) FROM voice_templates WHERE business_account_id=%s AND is_active=TRUE', (business_account_id,)) or 0),
        }
        return counts

    def _queue_health_counts(self, business_account_id: int) -> dict[str, int]:
        return {
            'outbound_failed': int(self._scalar("SELECT COUNT(*) FROM outbound_jobs WHERE business_account_id=%s AND status='failed'", (business_account_id,)) or 0),
            'outbound_overdue': int(self._scalar(
                "SELECT COUNT(*) FROM outbound_jobs WHERE business_account_id=%s AND status IN ('queued','sending') AND send_after < NOW() - INTERVAL '15 minutes'",
                (business_account_id,),
            ) or 0),
            'followups_open': int(self._scalar(
                "SELECT COUNT(*) FROM proactive_followup_plans WHERE business_account_id=%s AND status IN ('scheduled','processing')",
                (business_account_id,),
            ) or 0),
            'followups_overdue': int(self._scalar(
                "SELECT COUNT(*) FROM proactive_followup_plans WHERE business_account_id=%s AND status IN ('scheduled','processing') AND scheduled_for < NOW() - INTERVAL '30 minutes'",
                (business_account_id,),
            ) or 0),
            'admin_queue_open': int(self._scalar(
                "SELECT COUNT(*) FROM admin_queue WHERE business_account_id=%s AND status='open'",
                (business_account_id,),
            ) or 0),
            'pending_receipts': int(self._scalar(
                "SELECT COUNT(*) FROM receipts WHERE business_account_id=%s AND status='pending'",
                (business_account_id,),
            ) or 0),
        }

    def run_report(self, business_account_id: int | None = None, trigger_source: str = 'manual') -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        env_missing: list[str] = []
        if not self.settings.database_url:
            env_missing.append('DATABASE_URL')
        if not self.settings.tg_bot_token:
            env_missing.append('TG_BOT_TOKEN')
        if not self.settings.openai_api_key:
            env_missing.append('OPENAI_API_KEY')
        if not self.settings.telegram_webhook_url:
            env_missing.append('TELEGRAM_WEBHOOK_URL')
        if not self.settings.telegram_webhook_secret_token:
            env_missing.append('TELEGRAM_WEBHOOK_SECRET_TOKEN')
        if env_missing:
            self._append(checks, 'env.required', 'fail', '缺少关键环境变量。', {'missing': env_missing})
        else:
            self._append(checks, 'env.required', 'pass', '关键环境变量已配置。', {'webhook_url': self.settings.telegram_webhook_url})
        if self.settings.telegram_webhook_url and not self.settings.telegram_webhook_url.startswith('https://'):
            self._append(checks, 'env.webhook_https', 'warn', 'Webhook URL 不是 HTTPS，生产部署可能失败。', {'url': self.settings.telegram_webhook_url})
        else:
            self._append(checks, 'env.webhook_https', 'pass', 'Webhook URL 使用 HTTPS。')
        if not self.settings.admin_chat_ids:
            self._append(checks, 'env.admin_ids', 'warn', '未配置管理员 chat id，任何私聊用户都可能进入管理端。')
        else:
            self._append(checks, 'env.admin_ids', 'pass', '管理员 chat id 已配置。', {'count': len(self.settings.admin_chat_ids)})

        missing_tables: list[str] = []
        for table_name in self.REQUIRED_TABLES:
            if not self._table_exists(table_name):
                missing_tables.append(table_name)
        if missing_tables:
            self._append(checks, 'db.tables', 'fail', '存在缺失的数据表。', {'missing_tables': missing_tables})
        else:
            self._append(checks, 'db.tables', 'pass', '关键数据表已就位。', {'count': len(self.REQUIRED_TABLES)})

        missing_user_columns = [col for col in self.REQUIRED_USER_COLUMNS if not self._column_exists('users', col)]
        if missing_user_columns:
            self._append(checks, 'db.users_columns', 'fail', 'users 表缺少关键语言字段。', {'missing_columns': missing_user_columns})
        else:
            self._append(checks, 'db.users_columns', 'pass', 'users 语言字段完整。')

        default_account = self.business_account_repo.get_default_account()
        if business_account_id is None and default_account:
            business_account_id = int(default_account['id'])
        if business_account_id is None:
            self._append(checks, 'business.default_account', 'warn', '当前还没有 business account，首次收到业务消息后会自动创建。')
        else:
            account_row = self.business_account_repo.get_by_id(business_account_id)
            if not account_row:
                self._append(checks, 'business.default_account', 'fail', '指定的 business account 不存在。', {'business_account_id': business_account_id})
                business_account_id = None
            else:
                self._append(checks, 'business.default_account', 'pass', 'business account 已就位。', {'business_account_id': business_account_id, 'display_name': account_row.get('display_name')})

        content_counts: dict[str, int] = {}
        queue_counts: dict[str, int] = {}
        if business_account_id is not None:
            try:
                policy = self.runtime_policy_repo.get_policy(business_account_id)
                weekdays = self._parse_jsonish(policy.get('active_weekdays_json'), [1, 2, 3, 4, 5, 6, 7])
                window_ok = (policy.get('reply_window_start_hour') is not None and policy.get('reply_window_end_hour') is not None)
                if window_ok and str(policy.get('timezone_name') or '').strip():
                    self._append(checks, 'runtime.policy', 'pass', '工作时段策略已存在。', {'timezone': policy.get('timezone_name'), 'weekdays': weekdays})
                else:
                    self._append(checks, 'runtime.policy', 'warn', '工作时段策略存在，但时间窗配置不完整。', {'policy': policy})
            except Exception as exc:
                self._append(checks, 'runtime.policy', 'fail', '无法读取工作时段策略。', {'error': str(exc)})

            content_counts = self._content_counts(business_account_id)
            missing_content = [key for key, value in content_counts.items() if value <= 0 and key != 'voice_templates_active']
            if missing_content:
                self._append(checks, 'content.minimum', 'warn', '关键内容库仍有空缺，建议补齐后再正式部署。', {'counts': content_counts, 'missing': missing_content})
            else:
                self._append(checks, 'content.minimum', 'pass', '关键内容库已经具备基础投放能力。', {'counts': content_counts})
            if content_counts.get('voice_templates_active', 0) <= 0:
                self._append(checks, 'voice.templates', 'warn', '语音模板库为空，语音投递将不可用。', {'voice_templates_active': 0})
            else:
                self._append(checks, 'voice.templates', 'pass', '语音模板库已配置。', {'voice_templates_active': content_counts.get('voice_templates_active', 0)})

            queue_counts = self._queue_health_counts(business_account_id)
            if queue_counts['outbound_failed'] > 0:
                self._append(checks, 'queue.outbound_failed', 'warn', '存在失败的发送任务，建议部署前清理或排查。', queue_counts)
            else:
                self._append(checks, 'queue.outbound_failed', 'pass', '发送队列无失败任务。', queue_counts)
            if queue_counts['outbound_overdue'] > 0 or queue_counts['followups_overdue'] > 0:
                self._append(checks, 'queue.overdue', 'warn', '存在超时未处理的发送或跟进任务。', queue_counts)
            else:
                self._append(checks, 'queue.overdue', 'pass', '发送队列和主动跟进计划无严重堆积。', queue_counts)

            msg_counts = self._fetchone(
                '''
                SELECT
                    COUNT(*) FILTER (WHERE sender_type='user') AS user_messages,
                    COUNT(*) FILTER (WHERE sender_type='ai') AS ai_messages,
                    COUNT(*) FILTER (WHERE sender_type='human') AS human_messages
                FROM messages
                WHERE business_account_id=%s
                ''',
                (business_account_id,),
            )
            if int(msg_counts.get('user_messages') or 0) <= 0:
                self._append(checks, 'traffic.messages', 'warn', '当前账号还没有用户消息样本，建议先小规模真实跑通。', msg_counts)
            else:
                self._append(checks, 'traffic.messages', 'pass', '已有消息样本，可进行部署前联调。', msg_counts)

        overall_status = self._compute_overall_status(checks)
        readiness_score = self._compute_readiness_score(checks)
        summary = {
            'business_account_id': business_account_id,
            'overall_status': overall_status,
            'readiness_score': readiness_score,
            'pass_count': sum(1 for item in checks if item['status'] == 'pass'),
            'warn_count': sum(1 for item in checks if item['status'] == 'warn'),
            'fail_count': sum(1 for item in checks if item['status'] == 'fail'),
            'content_counts': content_counts,
            'queue_counts': queue_counts,
            'generated_at': utc_now().isoformat(),
            'trigger_source': trigger_source,
        }
        report_id = self.preflight_repo.create_report(business_account_id, trigger_source, overall_status, readiness_score, summary, checks)
        summary['report_id'] = report_id
        return {
            'ok': overall_status != 'fail',
            'report_id': report_id,
            'business_account_id': business_account_id,
            'overall_status': overall_status,
            'readiness_score': readiness_score,
            'summary': summary,
            'checks': checks,
        }

    def get_latest_report(self, business_account_id: int | None = None) -> dict[str, Any] | None:
        row = self.preflight_repo.get_latest_report(business_account_id)
        if not row:
            return None
        summary = row.get('summary_json') or {}
        checks = row.get('checks_json') or []
        return {
            'ok': str(row.get('overall_status') or '').lower() != 'fail',
            'report_id': row.get('id'),
            'business_account_id': row.get('business_account_id'),
            'overall_status': row.get('overall_status'),
            'readiness_score': row.get('readiness_score'),
            'summary': summary,
            'checks': checks,
            'created_at': row.get('created_at').isoformat() if row.get('created_at') else None,
        }

    @staticmethod
    def format_admin_text(report: dict[str, Any]) -> str:
        if not report:
            return '🧪 部署前检测\n\n暂无检测结果。'
        summary = report.get('summary') or {}
        checks = list(report.get('checks') or [])
        overall_status = str(report.get('overall_status') or 'unknown').lower()
        status_label = {'pass': '✅ 通过', 'warn': '⚠️ 有警告', 'fail': '❌ 不通过'}.get(overall_status, overall_status)
        score = int(report.get('readiness_score') or 0)
        lines = [
            '🧪 部署前检测',
            f'总体状态：{status_label}',
            f'就绪评分：{score}/100',
            f"通过 {summary.get('pass_count', 0)} | 警告 {summary.get('warn_count', 0)} | 失败 {summary.get('fail_count', 0)}",
        ]
        business_account_id = summary.get('business_account_id')
        if business_account_id:
            lines.append(f'账号ID：{business_account_id}')
        lines.append('')
        important = [item for item in checks if str(item.get('status')) in ('fail', 'warn')][:8]
        if not important:
            important = checks[:5]
        lines.append('重点结果：')
        for item in important:
            icon = {'pass': '✅', 'warn': '⚠️', 'fail': '❌'}.get(str(item.get('status')), '•')
            lines.append(f"{icon} {item.get('message')}")
        content_counts = summary.get('content_counts') or {}
        if content_counts:
            lines.append('')
            lines.append('内容概况：')
            counts = {k: int(v or 0) for k, v in content_counts.items()}
            lines.append(
                '项目 {projects_active} / 阶段 {segments_active} / 营销 {marketing_active} / 人设 {persona_materials_active} / 日常 {daily_materials_active} / 项目素材 {project_materials_active} / 素材包 {material_packages_active} / 语音 {voice_templates_active}'.format(**counts)
            )
        queue_counts = summary.get('queue_counts') or {}
        if queue_counts:
            lines.append('')
            lines.append('队列概况：')
            lines.append(
                f"失败发送 {int(queue_counts.get('outbound_failed') or 0)} / 超时发送 {int(queue_counts.get('outbound_overdue') or 0)} / 开放跟进 {int(queue_counts.get('followups_open') or 0)} / 超时跟进 {int(queue_counts.get('followups_overdue') or 0)}"
            )
        lines.append('')
        lines.append('建议：先处理失败项，再清理警告项，然后进入最终部署。')
        return '\n'.join(lines)


class TGAdminMenuBuilder(_Step10BaseTGAdminMenuBuilder):
    @staticmethod
    def main_menu() -> dict:
        menu = copy.deepcopy(_Step10BaseTGAdminMenuBuilder.main_menu())
        reply_markup = list(menu.get('reply_markup') or [])
        preflight_row = [{"text": "🧪 部署前检测", "callback_data": "adm:preflight"}]
        if preflight_row not in reply_markup:
            reply_markup.append(preflight_row)
        menu['reply_markup'] = reply_markup
        return menu


class TGAdminCallbackRouter(_Step10BaseTGAdminCallbackRouter):
    def __init__(self, admin_api_service: AdminAPIService, dashboard_service: DashboardService, tg_sender,
                 business_account_repo: BusinessAccountRepository, admin_session_repo: AdminSessionRepository,
                 preflight_service: DeploymentPreflightService | None = None) -> None:
        super().__init__(admin_api_service, dashboard_service, tg_sender, business_account_repo, admin_session_repo)
        self.preflight_service = preflight_service

    def handle(self, admin_chat_id: int, callback_data: str, operator: str = 'admin') -> None:
        parts = (callback_data or '').split(':')
        if parts[:2] == ['adm', 'preflight']:
            if not self.preflight_service:
                self._send_text(admin_chat_id, '🧪 部署前检测服务尚未初始化。', [[{"text": "🏠 主菜单", "callback_data": "adm:main"}]])
                return
            business_account_id = self._resolve_business_account_id()
            if len(parts) >= 3 and parts[2] == 'latest':
                report = self.preflight_service.get_latest_report(business_account_id)
                if not report:
                    report = self.preflight_service.run_report(business_account_id, trigger_source='manual')
            else:
                report = self.preflight_service.run_report(business_account_id, trigger_source='manual')
            text_value = self.preflight_service.format_admin_text(report)
            self._send_text(admin_chat_id, text_value, [
                [{"text": "🔄 重新检测", "callback_data": "adm:preflight"}, {"text": "📄 最新结果", "callback_data": "adm:preflight:latest"}],
                [{"text": "🏠 主菜单", "callback_data": "adm:main"}],
            ])
            return
        return super().handle(admin_chat_id, callback_data, operator)


def build_admin_blueprint(admin_api_service: AdminAPIService, dashboard_service: DashboardService) -> Blueprint:
    bp = _Step10BaseBuildAdminBlueprint(admin_api_service, dashboard_service)
    preflight_service = _STEP10_PREFLIGHT_SERVICE

    @bp.get('/preflight/latest')
    def preflight_latest():
        if preflight_service is None:
            return jsonify({'ok': False, 'error': 'preflight service unavailable'}), 503
        business_account_id = request.args.get('business_account_id', default=None, type=int)
        data = preflight_service.get_latest_report(business_account_id)
        if not data:
            return jsonify({'ok': False, 'error': 'no report found'}), 404
        return jsonify({'ok': True, 'data': data})

    @bp.post('/preflight/run')
    def preflight_run():
        if preflight_service is None:
            return jsonify({'ok': False, 'error': 'preflight service unavailable'}), 503
        payload = request.get_json(silent=True) or {}
        business_account_id = payload.get('business_account_id')
        if business_account_id is not None:
            try:
                business_account_id = int(business_account_id)
            except Exception:
                return jsonify({'ok': False, 'error': 'business_account_id must be int'}), 400
        data = preflight_service.run_report(business_account_id, trigger_source='api')
        return jsonify({'ok': True, 'data': data})

    @bp.get('/preflight/history')
    def preflight_history():
        if preflight_service is None:
            return jsonify({'ok': False, 'error': 'preflight service unavailable'}), 503
        business_account_id = request.args.get('business_account_id', default=None, type=int)
        limit = request.args.get('limit', default=10, type=int)
        data = preflight_service.preflight_repo.list_recent_reports(business_account_id, limit=max(1, min(limit, 50)))
        return jsonify({'ok': True, 'data': data})

    return bp


def build_app_components(settings: Settings) -> dict[str, Any]:
    global _STEP10_PREFLIGHT_SERVICE
    app_components = _Step10BaseBuildAppComponents(settings)
    db = app_components['db']
    business_account_repo = BusinessAccountRepository(db)
    admin_session_repo = AdminSessionRepository(db)
    runtime_policy_repo = RuntimePolicyRepository(db, settings.default_timezone)
    preflight_repo = DeploymentPreflightRepository(db)
    preflight_service = DeploymentPreflightService(settings, db, business_account_repo, runtime_policy_repo, preflight_repo)
    _STEP10_PREFLIGHT_SERVICE = preflight_service

    tg_client = app_components['tg_client']
    admin_api_service = app_components['admin_api_service']
    dashboard_service = app_components['dashboard_service']
    tg_admin_callback_router = TGAdminCallbackRouter(admin_api_service, dashboard_service, tg_client, business_account_repo, admin_session_repo, preflight_service)
    tg_admin_handlers = TGAdminHandlers(tg_admin_callback_router, tg_client, admin_session_repo, business_account_repo, admin_api_service)

    app_components['deployment_preflight_service'] = preflight_service
    app_components['tg_admin_callback_router'] = tg_admin_callback_router
    app_components['tg_admin_handlers'] = tg_admin_handlers
    return app_components


def main() -> None:
    settings = Settings.load()
    settings.validate()
    setup_logging(settings.log_level)
    logger.info('Starting application.')
    app_components = build_app_components(settings)

    preflight_service = app_components.get('deployment_preflight_service')
    if preflight_service is not None:
        try:
            startup_report = preflight_service.run_report(trigger_source='startup')
            logger.info(
                'Deployment preflight | status=%s | readiness_score=%s | pass=%s | warn=%s | fail=%s',
                startup_report.get('overall_status'),
                startup_report.get('readiness_score'),
                (startup_report.get('summary') or {}).get('pass_count'),
                (startup_report.get('summary') or {}).get('warn_count'),
                (startup_report.get('summary') or {}).get('fail_count'),
            )
        except Exception:
            logger.exception('Deployment preflight failed during startup.')

    outbound_sender_worker = app_components.get('outbound_sender_worker')
    if outbound_sender_worker is not None:
        outbound_sender_worker.start()

    tg_client = app_components['tg_client']
    if settings.telegram_webhook_url:
        try:
            tg_client.ensure_webhook(
                webhook_url=settings.telegram_webhook_url,
                secret_token=settings.telegram_webhook_secret_token,
            )
            logger.info('Telegram webhook ensured.')
        except Exception:
            logger.exception('Failed to ensure Telegram webhook.')

    flask_app = create_web_app(settings, app_components)
    logger.info('Application initialized successfully | host=%s | port=%s', settings.webhook_host, settings.webhook_port)
    flask_app.run(host=settings.webhook_host, port=settings.webhook_port)



# =========================
# step11 hardening overrides
# =========================

_Step10BaseSettings = Settings
_Step10BaseDatabase = Database
_Step10CurrentBuildAdminBlueprint = build_admin_blueprint
_Step10CurrentBuildAppComponents = build_app_components
_Step10CurrentCreateWebApp = create_web_app

_STEP11_SETTINGS: Settings | None = None
_STEP11_RUNTIME_LOCK = threading.Lock()
_STEP11_RUNTIME_COMPONENTS: dict[str, Any] | None = None


@dataclass
class Settings(_Step10BaseSettings):
    web_admin_token: str = ""

    @classmethod
    def load(cls) -> "Settings":
        base = _Step10BaseSettings.load()
        port_value = os.getenv("PORT", os.getenv("WEBHOOK_PORT", str(base.webhook_port or 8080))).strip() or "8080"
        try:
            resolved_port = int(port_value)
        except Exception:
            resolved_port = 8080
        return cls(**{**base.__dict__, "webhook_port": resolved_port, "web_admin_token": os.getenv("WEB_ADMIN_TOKEN", "").strip()})

    def validate(self) -> None:
        super().validate()
        missing: list[str] = []
        if not self.admin_chat_ids:
            missing.append("ADMIN_CHAT_IDS")
        if not self.web_admin_token:
            missing.append("WEB_ADMIN_TOKEN")
        if self.telegram_webhook_url and not self.telegram_webhook_url.startswith("https://"):
            missing.append("TELEGRAM_WEBHOOK_URL(https required)")
        if missing:
            raise RuntimeError(f"Missing or invalid required environment variables: {', '.join(missing)}")


class Database:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: dict[int, psycopg.Connection] = {}

    def _create_connection(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn, autocommit=False, row_factory=dict_row)

    def _get_connection(self) -> psycopg.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None or conn.closed:
            conn = self._create_connection()
            self._local.conn = conn
            with self._lock:
                self._connections[threading.get_ident()] = conn
        return conn

    def connect(self) -> None:
        self._get_connection()

    def close(self) -> None:
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None
                with self._lock:
                    self._connections.pop(threading.get_ident(), None)

    def close_all(self) -> None:
        with self._lock:
            conns = list(self._connections.items())
            self._connections.clear()
        for thread_id, conn in conns:
            try:
                conn.close()
            except Exception:
                logger.exception('Failed to close db connection | thread_id=%s', thread_id)
        self._local.conn = None

    def rollback(self) -> None:
        conn = getattr(self._local, 'conn', None)
        if conn is not None and not conn.closed:
            conn.rollback()

    def commit(self) -> None:
        conn = getattr(self._local, 'conn', None)
        if conn is not None and not conn.closed:
            conn.commit()

    @contextmanager
    def cursor(self):
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    @contextmanager
    def transaction(self):
        conn = self._get_connection()
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _is_admin_chat_allowed(settings: Settings, chat_id: int) -> bool:
    return bool(settings.admin_chat_ids and chat_id in settings.admin_chat_ids)


def _verify_web_admin_request(settings: Settings) -> tuple[bool, tuple[dict[str, Any], int] | None]:
    presented = (request.headers.get('X-Admin-Token') or '').strip()
    auth_header = (request.headers.get('Authorization') or '').strip()
    if not presented and auth_header.lower().startswith('bearer '):
        presented = auth_header[7:].strip()
    if not presented:
        return False, ({'ok': False, 'error': 'admin auth required'}, 401)
    if not hmac.compare_digest(presented, settings.web_admin_token):
        return False, ({'ok': False, 'error': 'admin auth invalid'}, 403)
    return True, None


def build_admin_blueprint(admin_api_service: AdminAPIService, dashboard_service: DashboardService) -> Blueprint:
    bp = _Step10CurrentBuildAdminBlueprint(admin_api_service, dashboard_service)
    settings = _STEP11_SETTINGS

    @bp.before_request
    def _authenticate_admin_routes():
        if settings is None:
            return jsonify({'ok': False, 'error': 'admin settings unavailable'}), 503
        ok, error = _verify_web_admin_request(settings)
        if not ok:
            payload, status = error
            return jsonify(payload), status
        return None

    return bp


def build_app_components(settings: Settings) -> dict[str, Any]:
    global _STEP11_SETTINGS
    _STEP11_SETTINGS = settings
    app_components = _Step10CurrentBuildAppComponents(settings)
    db = app_components['db']
    app_components['business_account_repo'] = BusinessAccountRepository(db)
    app_components['event_lock_repo'] = WebhookEventLockRepository(db)
    app_components['business_message_edit_repo'] = BusinessMessageEditRepository(db)
    app_components['admin_session_repo'] = AdminSessionRepository(db)
    return app_components


def create_web_app(settings: Settings, app_components: dict[str, Any]) -> Flask:
    flask_app = Flask(__name__)
    admin_api_service = app_components['admin_api_service']
    dashboard_service = app_components['dashboard_service']
    gateway = app_components['gateway']
    tg_admin_handlers = app_components['tg_admin_handlers']
    tg_client = app_components['tg_client']
    event_lock_repo: WebhookEventLockRepository = app_components['event_lock_repo']
    business_account_repo: BusinessAccountRepository = app_components['business_account_repo']
    db: Database = app_components['db']

    flask_app.register_blueprint(build_admin_blueprint(admin_api_service, dashboard_service))

    @flask_app.teardown_appcontext
    def _close_request_db(_exc=None):
        try:
            db.close()
        except Exception:
            logger.exception('Failed to close request db connection.')

    @flask_app.get('/health')
    def health():
        return jsonify({'ok': True, 'service': 'tg-business-ai-chat'})

    @flask_app.post('/webhook/telegram')
    def telegram_webhook():
        raw_update = request.get_json(silent=True) or {}
        if settings.telegram_webhook_secret_token:
            header_token = request.headers.get('X-Telegram-Bot-Api-Secret-Token', '')
            if not hmac.compare_digest(header_token, settings.telegram_webhook_secret_token):
                return jsonify({'ok': False, 'error': 'invalid webhook secret'}), 403
        try:
            event = normalize_telegram_update(raw_update)
            if event and event.event_type in ('admin_callback', 'admin_text'):
                admin_chat_id = int(event.telegram_chat_id or 0)
                if not _is_admin_chat_allowed(settings, admin_chat_id):
                    return jsonify({'ok': False, 'error': 'admin not allowed'}), 403
                event.idempotency_key = build_event_idempotency_key(event)
                account = business_account_repo.resolve_from_connection(event.business_connection_id)
                event.business_account_id = None if not account else int(account['id'])
                locked = event_lock_repo.acquire(
                    event.idempotency_key,
                    event.event_type,
                    event.business_account_id,
                    event.business_connection_id,
                    event.telegram_chat_id,
                    event.telegram_user_id,
                    event.telegram_message_id,
                    event.telegram_update_id,
                    event.is_edited,
                )
                if not locked:
                    event_lock_repo.record_duplicate_skip(event.idempotency_key)
                    if event.event_type == 'admin_callback' and raw_update.get('callback_query', {}).get('id'):
                        tg_client.answer_callback_query(str(raw_update['callback_query']['id']))
                    return jsonify({'ok': True, 'dispatched': 'duplicate'})
                try:
                    operator = str((raw_update.get('callback_query') or {}).get('from', {}).get('id') or (raw_update.get('message') or {}).get('from', {}).get('id') or 'admin')
                    if event.event_type == 'admin_callback':
                        tg_admin_handlers.handle_admin_callback(admin_chat_id=admin_chat_id, callback_data=event.callback_data or '', operator=operator)
                        callback_id = (raw_update.get('callback_query') or {}).get('id')
                        if callback_id:
                            tg_client.answer_callback_query(str(callback_id))
                    else:
                        text = str(event.text or '')
                        if text.startswith('/admin') or text.startswith('/startadmin'):
                            tg_admin_handlers.handle_admin_command(admin_chat_id, text, operator=operator)
                            dispatched = 'admin_command'
                        else:
                            tg_admin_handlers.handle_admin_text(admin_chat_id, text, operator=operator)
                            dispatched = 'admin_text'
                        event_lock_repo.mark_processed(event.idempotency_key)
                        return jsonify({'ok': True, 'dispatched': dispatched})
                    event_lock_repo.mark_processed(event.idempotency_key)
                    return jsonify({'ok': True, 'dispatched': event.event_type})
                except Exception as admin_exc:
                    event_lock_repo.mark_failed(event.idempotency_key, str(admin_exc))
                    raise
            dispatched = gateway.handle_raw_update(raw_update)
            return jsonify({'ok': True, 'dispatched': dispatched})
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            logger.exception('telegram_webhook failed')
            return jsonify({'ok': False, 'error': str(e)}), 500

    return flask_app


def _ensure_webhook_once(settings: Settings, app_components: dict[str, Any]) -> None:
    tg_client = app_components['tg_client']
    if settings.telegram_webhook_url:
        try:
            tg_client.ensure_webhook(webhook_url=settings.telegram_webhook_url, secret_token=settings.telegram_webhook_secret_token)
            logger.info('Telegram webhook ensured.')
        except Exception:
            logger.exception('Failed to ensure Telegram webhook.')


def _run_startup_preflight(app_components: dict[str, Any]) -> None:
    preflight_service = app_components.get('deployment_preflight_service')
    if preflight_service is None:
        return
    try:
        startup_report = preflight_service.run_report(trigger_source='startup')
        logger.info(
            'Deployment preflight | status=%s | readiness_score=%s | pass=%s | warn=%s | fail=%s',
            startup_report.get('overall_status'),
            startup_report.get('readiness_score'),
            (startup_report.get('summary') or {}).get('pass_count'),
            (startup_report.get('summary') or {}).get('warn_count'),
            (startup_report.get('summary') or {}).get('fail_count'),
        )
    except Exception:
        logger.exception('Deployment preflight failed during startup.')


def initialize_runtime(settings: Settings) -> dict[str, Any]:
    global _STEP11_RUNTIME_COMPONENTS
    with _STEP11_RUNTIME_LOCK:
        if _STEP11_RUNTIME_COMPONENTS is None:
            app_components = build_app_components(settings)
            outbound_sender_worker = app_components.get('outbound_sender_worker')
            if outbound_sender_worker is not None:
                outbound_sender_worker.start()
            _run_startup_preflight(app_components)
            _ensure_webhook_once(settings, app_components)
            _STEP11_RUNTIME_COMPONENTS = app_components
        return _STEP11_RUNTIME_COMPONENTS


def create_production_app() -> Flask:
    settings = Settings.load()
    settings.validate()
    setup_logging(settings.log_level)
    logger.info('Starting production app factory.')
    app_components = initialize_runtime(settings)
    flask_app = create_web_app(settings, app_components)
    return flask_app


def main() -> None:
    settings = Settings.load()
    settings.validate()
    setup_logging(settings.log_level)
    logger.info('Starting application.')
    app_components = initialize_runtime(settings)
    flask_app = create_web_app(settings, app_components)
    logger.info('Application initialized successfully | host=%s | port=%s', settings.webhook_host, settings.webhook_port)
    flask_app.run(host=settings.webhook_host, port=settings.webhook_port, threaded=True)


if __name__ == '__main__':
    main()
