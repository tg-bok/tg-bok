from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import asyncpg
import httpx
from flask import Flask, jsonify, request


# =========================================================
# Configuration & constants
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook/telegram")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
DEFAULT_BUSINESS_ACCOUNT_ID = int(os.getenv("DEFAULT_BUSINESS_ACCOUNT_ID", "1"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MAX_RECENT_MESSAGES = 30
MAX_MEMORY_ITEMS = 20
MAX_MATERIAL_MATCHES = 8
MAX_MARKETING_SEGMENTS = 5


# =========================================================
# Utility helpers
# =========================================================


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def safe_json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def compact_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# Dataclasses / schemas
# =========================================================

@dataclass(slots=True)
class UnderstandingResult:
    explicit_intent: str = ""
    hidden_intent: str = ""
    emotion_state: str = "neutral"
    boundary_signal: str = "none"
    busy_signal: bool = False
    interest_signal: str = "unknown"
    hesitation_level: str = "unknown"
    readiness_level: str = "unknown"
    style_signal: str = "balanced"
    topic_tags: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass(slots=True)
class StageDecision:
    stage_key: str = "initial"
    stage_confidence: float = 0.0
    trust_level: str = "low"
    engagement_level: str = "low"
    should_hold: bool = False
    should_progress: bool = False
    should_pause: bool = False
    reason: str = ""


@dataclass(slots=True)
class TurnDecision:
    turn_goal: str = "respond"
    relationship_mode: str = "neutral"
    project_mode: str = "none"
    initiative_level: str = "balanced"
    question_rate: str = "low"
    reply_length: str = "medium"
    marketing_intensity: int = 0
    should_send_media: bool = False
    preferred_media_types: list[str] = field(default_factory=list)
    should_follow_up_later: bool = False
    should_end_softly: bool = False
    reason: str = ""


@dataclass(slots=True)
class StyleSpec:
    warmth_level: str = "medium"
    formality_level: str = "medium"
    distance_level: str = "balanced"
    initiative_tone: str = "balanced"
    question_style: str = "light"
    business_tone: str = "light_business"
    life_tone: str = "natural"
    maturity_tone: str = "mature"
    emoji_level: str = "low"
    cadence_note: str = "natural"


@dataclass(slots=True)
class UserStateSnapshot:
    user_id: int
    business_account_id: int
    current_project_id: int | None = None
    project_locked: bool = False
    tags: list[str] = field(default_factory=list)
    tags_locked: bool = False
    classification: str = "unknown"
    operation_classification: str = "unknown"
    intent_level: str = "unknown"
    is_under_handover: bool = False
    recent_summary: str = ""
    trust_score: float = 0.0
    last_stage: str = "initial"
    last_trajectory: str = "unknown"
    last_rhythm: str = "unknown"
    last_maturity: str = "unknown"


@dataclass(slots=True)
class MaterialMatchResult:
    material_id: int
    material_scope: str
    material_type: str
    title: str = ""
    content_text: str | None = None
    file_url: str | None = None
    semantic_score: float = 0.0
    scene_score: float = 0.0
    final_score: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class MarketingPlanMatchResult:
    plan_id: int
    customer_stage_type: str
    intensity_level: int
    matched_customer_tags: list[str] = field(default_factory=list)
    matched_segments: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class FAQMatchResult:
    faq_id: int
    question: str
    answer: str
    category: str = ""
    score: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class PersonaProfileSnapshot:
    name: str = ""
    gender: str = "female"
    age: str = ""
    height: str = ""
    weight: str = ""
    nationality: str = ""
    current_residence: str = ""
    marital_status: str = ""
    has_children: str = ""
    employer: str = ""
    job_title: str = ""
    hobbies: str = ""
    financial_status: str = ""
    relationship_history: str = ""


# =========================================================
# Database and repositories
# =========================================================

class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            if not self._dsn:
                raise ValueError("DATABASE_URL is required")
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
            logger.info("Database pool initialized.")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database is not connected")
        return self._pool

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        return await self.pool.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        return await self.pool.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self.pool.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        return await self.pool.execute(query, *args)


class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    async def get_by_telegram_user_id(self, business_account_id: int, telegram_user_id: int) -> asyncpg.Record | None:
        return await self.db.fetchrow(
            "SELECT * FROM users WHERE business_account_id=$1 AND telegram_user_id=$2",
            business_account_id,
            telegram_user_id,
        )

    async def create_or_update_user(
        self,
        business_account_id: int,
        telegram_user_id: int,
        username: str | None,
        display_name: str | None,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO users (business_account_id, telegram_user_id, username, display_name)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (business_account_id, telegram_user_id)
            DO UPDATE SET username=EXCLUDED.username, display_name=EXCLUDED.display_name
            """,
            business_account_id,
            telegram_user_id,
            username,
            display_name,
        )

    async def set_project(self, user_id: int, project_id: int | None) -> None:
        await self.db.execute("UPDATE users SET current_project_id=$2 WHERE id=$1", user_id, project_id)

    async def set_tags(self, user_id: int, tags: list[str]) -> None:
        await self.db.execute("UPDATE users SET tags_json=$2 WHERE id=$1", user_id, safe_json_dumps(tags))

    async def set_classification(self, user_id: int, classification: str) -> None:
        await self.db.execute("UPDATE users SET classification=$2 WHERE id=$1", user_id, classification)

    async def set_operation_classification(self, user_id: int, operation_classification: str) -> None:
        await self.db.execute(
            "UPDATE users SET operation_classification=$2 WHERE id=$1",
            user_id,
            operation_classification,
        )

    async def set_project_lock(self, user_id: int, is_locked: bool) -> None:
        await self.db.execute("UPDATE users SET is_project_locked=$2 WHERE id=$1", user_id, is_locked)

    async def set_tags_lock(self, user_id: int, is_locked: bool) -> None:
        await self.db.execute("UPDATE users SET is_tags_locked=$2 WHERE id=$1", user_id, is_locked)

    async def set_handover_state(self, user_id: int, is_under_handover: bool) -> None:
        await self.db.execute("UPDATE users SET is_under_handover=$2 WHERE id=$1", user_id, is_under_handover)

    async def list_users_for_business_account(self, business_account_id: int, limit: int = 50) -> list[asyncpg.Record]:
        return await self.db.fetch(
            "SELECT * FROM users WHERE business_account_id=$1 ORDER BY id DESC LIMIT $2",
            business_account_id,
            limit,
        )

    async def get_user_by_id(self, user_id: int) -> asyncpg.Record | None:
        return await self.db.fetchrow("SELECT * FROM users WHERE id=$1", user_id)

    async def update_current_project(self, user_id: int, project_id: int | None) -> None:
        await self.set_project(user_id, project_id)

    async def update_tags(self, user_id: int, tags: list[str]) -> None:
        await self.set_tags(user_id, tags)

    async def update_classification(self, user_id: int, classification: str) -> None:
        await self.set_classification(user_id, classification)

    async def update_operation_classification(self, user_id: int, operation_classification: str) -> None:
        await self.set_operation_classification(user_id, operation_classification)


class ConversationRepository:
    def __init__(self, db: Database):
        self.db = db

    async def get_snapshot(self, business_account_id: int, user_id: int) -> UserStateSnapshot:
        user = await self.db.fetchrow("SELECT * FROM users WHERE id=$1", user_id)
        conv = await self.db.fetchrow(
            "SELECT * FROM conversations WHERE business_account_id=$1 AND user_id=$2",
            business_account_id,
            user_id,
        )
        if user is None:
            raise ValueError(f"User not found: {user_id}")
        return UserStateSnapshot(
            user_id=user_id,
            business_account_id=business_account_id,
            current_project_id=user.get("current_project_id"),
            project_locked=bool(user.get("is_project_locked", False)),
            tags=safe_json_loads(user.get("tags_json"), []),
            tags_locked=bool(user.get("is_tags_locked", False)),
            classification=user.get("classification") or "unknown",
            operation_classification=user.get("operation_classification") or "unknown",
            intent_level=user.get("intent_level") or "unknown",
            is_under_handover=bool(user.get("is_under_handover", False)),
            recent_summary=(conv.get("summary_text") if conv else "") or "",
            trust_score=float(conv.get("trust_score") if conv else 0.0),
            last_stage=(conv.get("last_stage") if conv else "initial") or "initial",
            last_trajectory=(conv.get("last_trajectory") if conv else "unknown") or "unknown",
            last_rhythm=(conv.get("last_rhythm") if conv else "unknown") or "unknown",
            last_maturity=(conv.get("last_maturity") if conv else "unknown") or "unknown",
        )

    async def upsert_state(
        self,
        business_account_id: int,
        user_id: int,
        *,
        last_stage: str,
        last_trajectory: str,
        last_rhythm: str,
        last_maturity: str,
        trust_score: float,
        summary_text: str,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO conversations (
                business_account_id, user_id, last_stage, last_trajectory, last_rhythm,
                last_maturity, trust_score, summary_text, last_ai_reply_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            ON CONFLICT (business_account_id, user_id)
            DO UPDATE SET
                last_stage=EXCLUDED.last_stage,
                last_trajectory=EXCLUDED.last_trajectory,
                last_rhythm=EXCLUDED.last_rhythm,
                last_maturity=EXCLUDED.last_maturity,
                trust_score=EXCLUDED.trust_score,
                summary_text=EXCLUDED.summary_text,
                last_ai_reply_at=EXCLUDED.last_ai_reply_at
            """,
            business_account_id,
            user_id,
            last_stage,
            last_trajectory,
            last_rhythm,
            last_maturity,
            trust_score,
            summary_text,
            utcnow(),
        )

    async def get_conversation_state(self, business_account_id: int, user_id: int) -> UserStateSnapshot:
        return await self.get_snapshot(business_account_id, user_id)


class MessageRepository:
    def __init__(self, db: Database):
        self.db = db

    async def create_message(
        self,
        business_account_id: int,
        user_id: int,
        role: str,
        content: str,
        content_type: str = "text",
        telegram_message_id: str | None = None,
        media_url: str | None = None,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO messages (
                business_account_id, user_id, role, content, content_type, telegram_message_id, media_url, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            business_account_id,
            user_id,
            role,
            content,
            content_type,
            telegram_message_id,
            media_url,
            utcnow(),
        )

    async def list_recent_messages(self, business_account_id: int, user_id: int, limit: int = MAX_RECENT_MESSAGES) -> list[asyncpg.Record]:
        return await self.db.fetch(
            """
            SELECT * FROM messages
            WHERE business_account_id=$1 AND user_id=$2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            business_account_id,
            user_id,
            limit,
        )


class ProjectRepository:
    def __init__(self, db: Database):
        self.db = db

    async def list_projects(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.db.fetch(
            "SELECT * FROM projects WHERE business_account_id=$1 ORDER BY id DESC",
            business_account_id,
        )

    async def get_project(self, project_id: int) -> asyncpg.Record | None:
        return await self.db.fetchrow("SELECT * FROM projects WHERE id=$1", project_id)

    async def create_project(self, business_account_id: int, project_key: str, project_name: str, description: str = "") -> None:
        await self.db.execute(
            "INSERT INTO projects (business_account_id, project_key, project_name, description, is_active, is_locked) VALUES ($1,$2,$3,$4,true,false)",
            business_account_id,
            project_key,
            project_name,
            description,
        )

    async def update_project(self, project_id: int, project_name: str, description: str) -> None:
        await self.db.execute(
            "UPDATE projects SET project_name=$2, description=$3 WHERE id=$1",
            project_id,
            project_name,
            description,
        )

    async def delete_project(self, project_id: int) -> None:
        await self.db.execute("DELETE FROM projects WHERE id=$1", project_id)

    async def set_lock(self, project_id: int, is_locked: bool) -> None:
        await self.db.execute("UPDATE projects SET is_locked=$2 WHERE id=$1", project_id, is_locked)

    async def list_projects_for_business_account(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.list_projects(business_account_id)

    async def get_project_by_id(self, project_id: int) -> asyncpg.Record | None:
        return await self.get_project(project_id)


class ProjectPromptRepository:
    def __init__(self, db: Database):
        self.db = db

    async def get_prompt(self, project_id: int) -> asyncpg.Record | None:
        return await self.db.fetchrow("SELECT * FROM project_prompts WHERE project_id=$1", project_id)

    async def upsert_prompt(self, project_id: int, guidance_text: str, applicable_customer_types: str = "", rhythm_note: str = "", boundary_note: str = "", caution_note: str = "") -> None:
        await self.db.execute(
            """
            INSERT INTO project_prompts (project_id, guidance_text, applicable_customer_types, rhythm_note, boundary_note, caution_note)
            VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT (project_id)
            DO UPDATE SET guidance_text=$2, applicable_customer_types=$3, rhythm_note=$4, boundary_note=$5, caution_note=$6
            """,
            project_id,
            guidance_text,
            applicable_customer_types,
            rhythm_note,
            boundary_note,
            caution_note,
        )

    async def get_prompt_for_project(self, project_id: int) -> asyncpg.Record | None:
        return await self.get_prompt(project_id)


class ProjectFAQRepository:
    def __init__(self, db: Database):
        self.db = db

    async def list_by_project(self, project_id: int) -> list[asyncpg.Record]:
        return await self.db.fetch("SELECT * FROM project_faqs WHERE project_id=$1 ORDER BY priority DESC, id DESC", project_id)

    async def create_faq(self, project_id: int, category: str, question: str, answer: str, priority: int = 0, is_enabled: bool = True, notes: str = "") -> None:
        await self.db.execute(
            "INSERT INTO project_faqs (project_id, category, question, answer, priority, is_enabled, notes) VALUES ($1,$2,$3,$4,$5,$6,$7)",
            project_id,
            category,
            question,
            answer,
            priority,
            is_enabled,
            notes,
        )

    async def list_faqs(self, project_id: int) -> list[asyncpg.Record]:
        return await self.list_by_project(project_id)


class MarketingPlanRepository:
    def __init__(self, db: Database):
        self.db = db

    async def list_by_project(self, project_id: int) -> list[asyncpg.Record]:
        return await self.db.fetch("SELECT * FROM marketing_plans WHERE project_id=$1 ORDER BY priority DESC, id DESC", project_id)

    async def list_by_stage_type(self, project_id: int, customer_stage_type: str) -> list[asyncpg.Record]:
        return await self.db.fetch(
            "SELECT * FROM marketing_plans WHERE project_id=$1 AND customer_stage_type=$2 ORDER BY priority DESC, id DESC",
            project_id,
            customer_stage_type,
        )

    async def create_plan(
        self,
        project_id: int,
        customer_stage_type: str,
        customer_tags_json: list[str],
        intensity_level: int,
        objective: str,
        strategy_note: str,
        priority: int = 0,
        is_enabled: bool = True,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO marketing_plans (
                project_id, customer_stage_type, customer_tags_json, intensity_level,
                objective, strategy_note, priority, is_enabled
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            project_id,
            customer_stage_type,
            safe_json_dumps(customer_tags_json),
            intensity_level,
            objective,
            strategy_note,
            priority,
            is_enabled,
        )

    async def list_plans(self, project_id: int, customer_stage_type: str | None = None) -> list[asyncpg.Record]:
        if customer_stage_type:
            return await self.list_by_stage_type(project_id, customer_stage_type)
        return await self.list_by_project(project_id)


class MarketingPlanSegmentRepository:
    def __init__(self, db: Database):
        self.db = db

    async def list_by_plan(self, marketing_plan_id: int) -> list[asyncpg.Record]:
        return await self.db.fetch(
            "SELECT * FROM marketing_plan_segments WHERE marketing_plan_id=$1 ORDER BY priority DESC, id DESC",
            marketing_plan_id,
        )

    async def create_segment(self, marketing_plan_id: int, segment_title: str, segment_type: str, content: str, priority: int = 0, is_enabled: bool = True) -> None:
        await self.db.execute(
            "INSERT INTO marketing_plan_segments (marketing_plan_id, segment_title, segment_type, content, priority, is_enabled) VALUES ($1,$2,$3,$4,$5,$6)",
            marketing_plan_id,
            segment_title,
            segment_type,
            content,
            priority,
            is_enabled,
        )

    async def list_segments(self, marketing_plan_id: int) -> list[asyncpg.Record]:
        return await self.list_by_plan(marketing_plan_id)


class MaterialRepository:
    def __init__(self, db: Database):
        self.db = db

    async def list_materials(self, business_account_id: int, material_scope: str, project_id: int | None = None) -> list[asyncpg.Record]:
        if project_id is None:
            return await self.db.fetch(
                "SELECT * FROM materials WHERE business_account_id=$1 AND material_scope=$2 ORDER BY priority DESC, id DESC",
                business_account_id,
                material_scope,
            )
        return await self.db.fetch(
            "SELECT * FROM materials WHERE business_account_id=$1 AND material_scope=$2 AND project_id=$3 ORDER BY priority DESC, id DESC",
            business_account_id,
            material_scope,
            project_id,
        )

    async def create_material(
        self,
        business_account_id: int,
        project_id: int | None,
        material_scope: str,
        material_type: str,
        title: str,
        description: str = "",
        content_text: str = "",
        file_url: str = "",
        semantic_tags_json: list[str] | None = None,
        keyword_tags_json: list[str] | None = None,
        scene_tags_json: list[str] | None = None,
        applicable_customer_tags_json: list[str] | None = None,
        applicable_stage_tags_json: list[str] | None = None,
        priority: int = 0,
        is_enabled: bool = True,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO materials (
                business_account_id, project_id, material_scope, material_type, title, description,
                content_text, file_url, semantic_tags_json, keyword_tags_json, scene_tags_json,
                applicable_customer_tags_json, applicable_stage_tags_json, priority, is_enabled
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
            """,
            business_account_id,
            project_id,
            material_scope,
            material_type,
            title,
            description,
            content_text,
            file_url,
            safe_json_dumps(semantic_tags_json or []),
            safe_json_dumps(keyword_tags_json or []),
            safe_json_dumps(scene_tags_json or []),
            safe_json_dumps(applicable_customer_tags_json or []),
            safe_json_dumps(applicable_stage_tags_json or []),
            priority,
            is_enabled,
        )


class PersonaProfileRepository:
    def __init__(self, db: Database):
        self.db = db

    async def get_active_profile(self, business_account_id: int) -> asyncpg.Record | None:
        return await self.db.fetchrow(
            "SELECT * FROM persona_profiles WHERE business_account_id=$1 AND is_active=true ORDER BY id DESC LIMIT 1",
            business_account_id,
        )

    async def upsert_active_profile(self, business_account_id: int, profile: PersonaProfileSnapshot) -> None:
        await self.db.execute(
            """
            INSERT INTO persona_profiles (
                business_account_id, name, gender, age, height, weight, nationality, current_residence,
                marital_status, has_children, employer, job_title, hobbies, financial_status,
                relationship_history, is_active
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,true)
            """,
            business_account_id,
            profile.name,
            profile.gender,
            profile.age,
            profile.height,
            profile.weight,
            profile.nationality,
            profile.current_residence,
            profile.marital_status,
            profile.has_children,
            profile.employer,
            profile.job_title,
            profile.hobbies,
            profile.financial_status,
            profile.relationship_history,
        )


class MemoryRepository:
    def __init__(self, db: Database):
        self.db = db

    async def list_memories(self, business_account_id: int, user_id: int, limit: int = MAX_MEMORY_ITEMS) -> list[asyncpg.Record]:
        return await self.db.fetch(
            "SELECT * FROM memories WHERE business_account_id=$1 AND user_id=$2 ORDER BY importance DESC, id DESC LIMIT $3",
            business_account_id,
            user_id,
            limit,
        )

    async def create_memory(
        self,
        business_account_id: int,
        user_id: int,
        memory_type: str,
        memory_key: str,
        memory_value: str,
        importance: float,
        source: str,
    ) -> None:
        await self.db.execute(
            "INSERT INTO memories (business_account_id, user_id, memory_type, memory_key, memory_value, importance, source) VALUES ($1,$2,$3,$4,$5,$6,$7)",
            business_account_id,
            user_id,
            memory_type,
            memory_key,
            memory_value,
            importance,
            source,
        )


class HandoverRepository:
    def __init__(self, db: Database):
        self.db = db

    async def start_handover(self, business_account_id: int, user_id: int) -> None:
        await self.db.execute(
            "INSERT INTO handovers (business_account_id, user_id, started_at, is_active) VALUES ($1,$2,$3,true)",
            business_account_id,
            user_id,
            utcnow(),
        )

    async def end_handover(self, handover_id: int, handover_summary: str = "", resume_suggestion: str = "") -> None:
        await self.db.execute(
            "UPDATE handovers SET ended_at=$2, handover_summary=$3, resume_suggestion=$4, is_active=false WHERE id=$1",
            handover_id,
            utcnow(),
            handover_summary,
            resume_suggestion,
        )

    async def list_active(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.db.fetch(
            "SELECT * FROM handovers WHERE business_account_id=$1 AND is_active=true ORDER BY id DESC",
            business_account_id,
        )


class ReceiptRepository:
    def __init__(self, db: Database):
        self.db = db

    async def create_receipt(
        self,
        business_account_id: int,
        user_id: int,
        receipt_type: str,
        result_json: dict[str, Any],
        reason_text: str,
        confidence_score: float,
        source_type: str,
        status: str = "pending",
    ) -> None:
        await self.db.execute(
            "INSERT INTO receipts (business_account_id, user_id, receipt_type, result_json, reason_text, confidence_score, source_type, status) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
            business_account_id,
            user_id,
            receipt_type,
            safe_json_dumps(result_json),
            reason_text,
            confidence_score,
            source_type,
            status,
        )

    async def list_pending(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.db.fetch(
            "SELECT * FROM receipts WHERE business_account_id=$1 AND status='pending' ORDER BY id DESC",
            business_account_id,
        )


class QueueRepository:
    def __init__(self, db: Database):
        self.db = db

    async def create_queue_item(self, business_account_id: int, user_id: int, queue_type: str, priority: int, reason: str, status: str = "open") -> None:
        await self.db.execute(
            "INSERT INTO queues (business_account_id, user_id, queue_type, priority, reason, status) VALUES ($1,$2,$3,$4,$5,$6)",
            business_account_id,
            user_id,
            queue_type,
            priority,
            reason,
            status,
        )

    async def list_by_type(self, business_account_id: int, queue_type: str) -> list[asyncpg.Record]:
        return await self.db.fetch(
            "SELECT * FROM queues WHERE business_account_id=$1 AND queue_type=$2 ORDER BY priority DESC, id DESC",
            business_account_id,
            queue_type,
        )


class AuditRepository:
    def __init__(self, db: Database):
        self.db = db

    async def create_audit_log(self, business_account_id: int, user_id: int | None, audit_type: str, payload: dict[str, Any]) -> None:
        await self.db.execute(
            "INSERT INTO audit_logs (business_account_id, user_id, audit_type, payload_json, created_at) VALUES ($1,$2,$3,$4,$5)",
            business_account_id,
            user_id,
            audit_type,
            safe_json_dumps(payload),
            utcnow(),
        )

# =========================================================
# Service layer
# =========================================================


class AdminAPIService:
    def __init__(
        self,
        user_service: "UserManagementService",
        project_service: "ProjectManagementService",
        faq_service: "ProjectFAQService",
        marketing_service: "MarketingPlanService",
        project_prompt_service: "ProjectPromptService",
        material_service: "MaterialService",
        persona_service: "PersonaProfileService",
        daily_material_service: "DailyMaterialService",
        handover_service: "HandoverService",
        receipt_service: "ReceiptService",
        queue_service: "QueueService",
        audit_service: "AuditService",
    ):
        self.user_service = user_service
        self.project_service = project_service
        self.faq_service = faq_service
        self.marketing_service = marketing_service
        self.project_prompt_service = project_prompt_service
        self.material_service = material_service
        self.persona_service = persona_service
        self.daily_material_service = daily_material_service
        self.handover_service = handover_service
        self.receipt_service = receipt_service
        self.queue_service = queue_service
        self.audit_service = audit_service

    async def get_user_list(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.user_service.list_users(business_account_id)

    async def get_user_detail(self, business_account_id: int, user_id: int) -> dict[str, Any]:
        return await self.user_service.get_user_detail(business_account_id, user_id)

    async def update_user_project(self, user_id: int, project_id: int | None) -> None:
        await self.user_service.set_project(user_id, project_id)

    async def update_user_tags(self, user_id: int, tags: list[str]) -> None:
        await self.user_service.set_tags(user_id, tags)

    async def get_projects(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.project_service.list_projects(business_account_id)

    async def get_project_detail(self, project_id: int) -> dict[str, Any]:
        return await self.project_service.get_project_detail(project_id)

    async def get_project_faqs(self, project_id: int) -> list[asyncpg.Record]:
        return await self.faq_service.list_faqs(project_id)

    async def get_marketing_plans(self, project_id: int, customer_stage_type: str | None = None) -> list[asyncpg.Record]:
        return await self.marketing_service.list_plans(project_id, customer_stage_type)

    async def get_materials(
        self,
        business_account_id: int,
        *,
        scope: str,
        project_id: int | None = None,
    ) -> list[asyncpg.Record]:
        if scope == "daily":
            return await self.daily_material_service.list_daily_materials(business_account_id)
        if scope == "persona":
            return await self.persona_service.list_persona_materials(business_account_id)
        return await self.material_service.list_project_materials(business_account_id, project_id)

    async def get_persona_profile(self, business_account_id: int) -> asyncpg.Record | None:
        return await self.persona_service.get_active_profile(business_account_id)

    async def get_active_handovers(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.handover_service.list_active(business_account_id)

    async def get_pending_receipts(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.receipt_service.list_pending(business_account_id)

    async def get_queue_items(self, business_account_id: int, queue_type: str) -> list[asyncpg.Record]:
        if queue_type == "all":
            return await self.queue_service.list_all(business_account_id)
        return await self.queue_service.list_by_type(business_account_id, queue_type)

    async def get_project_prompt(self, project_id: int) -> asyncpg.Record | None:
        return await self.project_prompt_service.get_prompt(project_id)

    async def get_marketing_segments(self, plan_id: int) -> list[asyncpg.Record]:
        return await self.marketing_service.list_segments(plan_id)

    async def get_persona_materials(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.persona_service.list_persona_materials(business_account_id)

    async def get_daily_materials(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.daily_material_service.list_daily_materials(business_account_id)

    async def start_handover(self, business_account_id: int, user_id: int) -> None:
        await self.handover_service.start(business_account_id, user_id)

    async def end_handover(self, business_account_id: int, user_id: int) -> None:
        await self.handover_service.end(business_account_id, user_id)


class DashboardService:
    def __init__(
        self,
        user_repo: UserRepository,
        handover_repo: HandoverRepository,
        receipt_repo: ReceiptRepository,
        queue_repo: QueueRepository,
    ):
        self.user_repo = user_repo
        self.handover_repo = handover_repo
        self.receipt_repo = receipt_repo
        self.queue_repo = queue_repo

    async def build_dashboard_summary(self, business_account_id: int) -> dict[str, Any]:
        users = await self.user_repo.list_users_for_business_account(business_account_id, limit=200)
        active_handovers = await self.handover_repo.list_active(business_account_id)
        pending_receipts = await self.receipt_repo.list_pending(business_account_id)
        high_intent = [u for u in users if (u.get("intent_level") if hasattr(u, 'get') else u["intent_level"]) in {"high", "very_high"}]
        return {
            "user_count": len(users),
            "high_intent_count": len(high_intent),
            "active_handover_count": len(active_handovers),
            "pending_receipt_count": len(pending_receipts),
        }


class UserManagementService:
    def __init__(
        self,
        user_repo: UserRepository,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        handover_repo: HandoverRepository,
        audit_repo: AuditRepository,
    ):
        self.user_repo = user_repo
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.handover_repo = handover_repo
        self.audit_repo = audit_repo

    async def list_users(self, business_account_id: int, limit: int = 50) -> list[asyncpg.Record]:
        return await self.user_repo.list_users_for_business_account(business_account_id, limit=limit)

    async def get_user_detail(self, business_account_id: int, user_id: int) -> dict[str, Any]:
        user = await self.user_repo.get_user_by_id(user_id)
        conversation = await self.conversation_repo.get_conversation_state(business_account_id, user_id)
        messages = await self.message_repo.list_recent_messages(business_account_id, user_id, limit=20)
        return {"user": user, "conversation": conversation, "recent_messages": messages}

    async def set_project(self, user_id: int, project_id: int | None) -> None:
        await self.user_repo.update_current_project(user_id, project_id)

    async def set_tags(self, user_id: int, tags: list[str]) -> None:
        await self.user_repo.update_tags(user_id, tags)

    async def set_classification(self, user_id: int, classification: str) -> None:
        await self.user_repo.update_classification(user_id, classification)

    async def set_operation_classification(self, user_id: int, operation_classification: str) -> None:
        await self.user_repo.update_operation_classification(user_id, operation_classification)

    async def lock_project(self, user_id: int, is_locked: bool) -> None:
        await self.user_repo.set_project_lock(user_id, is_locked)

    async def lock_tags(self, user_id: int, is_locked: bool) -> None:
        await self.user_repo.set_tags_lock(user_id, is_locked)


class ProjectManagementService:
    def __init__(
        self,
        project_repo: ProjectRepository,
        faq_repo: ProjectFAQRepository,
        marketing_plan_repo: MarketingPlanRepository,
        project_prompt_repo: ProjectPromptRepository,
        material_repo: MaterialRepository,
        user_repo: UserRepository,
    ):
        self.project_repo = project_repo
        self.faq_repo = faq_repo
        self.marketing_plan_repo = marketing_plan_repo
        self.project_prompt_repo = project_prompt_repo
        self.material_repo = material_repo
        self.user_repo = user_repo

    async def list_projects(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.project_repo.list_projects_for_business_account(business_account_id)

    async def get_project_detail(self, project_id: int) -> dict[str, Any]:
        project = await self.project_repo.get_project_by_id(project_id)
        if project is None:
            return {"project": None, "materials": [], "faqs": [], "marketing_plans": [], "users": []}
        business_account_id = project["business_account_id"]
        materials = await self.material_repo.list_materials(business_account_id=business_account_id, project_id=project_id, material_scope="project")
        faqs = await self.faq_repo.list_faqs(project_id)
        plans = await self.marketing_plan_repo.list_plans(project_id)
        users = [u for u in await self.user_repo.list_users_for_business_account(business_account_id, limit=500) if u.get("current_project_id") == project_id]
        return {"project": project, "materials": materials, "faqs": faqs, "marketing_plans": plans, "users": users}

    async def list_project_users(self, project_id: int) -> list[asyncpg.Record]:
        project = await self.project_repo.get_project_by_id(project_id)
        if project is None:
            return []
        users = await self.user_repo.list_users_for_business_account(project["business_account_id"], limit=500)
        return [u for u in users if u.get("current_project_id") == project_id]

    async def create_project(self, business_account_id: int, project_key: str, project_name: str, description: str) -> None:
        await self.project_repo.create_project(business_account_id, project_key, project_name, description)


class ProjectFAQService:
    def __init__(self, faq_repo: ProjectFAQRepository):
        self.faq_repo = faq_repo

    async def list_faqs(self, project_id: int) -> list[asyncpg.Record]:
        return await self.faq_repo.list_faqs(project_id)

    async def create_faq(self, project_id: int, category: str, question: str, answer: str, priority: int = 100) -> None:
        await self.faq_repo.create_faq(project_id, category, question, answer, priority)


class MarketingPlanService:
    def __init__(self, plan_repo: MarketingPlanRepository, segment_repo: MarketingPlanSegmentRepository):
        self.plan_repo = plan_repo
        self.segment_repo = segment_repo

    async def list_plans(self, project_id: int, customer_stage_type: str | None = None) -> list[asyncpg.Record]:
        return await self.plan_repo.list_plans(project_id, customer_stage_type)

    async def list_segments(self, marketing_plan_id: int) -> list[asyncpg.Record]:
        return await self.segment_repo.list_segments(marketing_plan_id)

    async def create_plan(
        self,
        project_id: int,
        customer_stage_type: str,
        customer_tags_json: list[str],
        intensity_level: int,
        objective: str,
        strategy_note: str,
        priority: int = 100,
    ) -> None:
        await self.plan_repo.create_plan(project_id, customer_stage_type, customer_tags_json, intensity_level, objective, strategy_note, priority)

    async def create_segment(self, marketing_plan_id: int, segment_title: str, segment_type: str, content: str, priority: int = 100) -> None:
        await self.segment_repo.create_segment(marketing_plan_id, segment_title, segment_type, content, priority)


class ProjectPromptService:
    def __init__(self, prompt_repo: ProjectPromptRepository):
        self.prompt_repo = prompt_repo

    async def get_prompt(self, project_id: int) -> asyncpg.Record | None:
        return await self.prompt_repo.get_prompt_for_project(project_id)

    async def upsert_prompt(
        self,
        project_id: int,
        guidance_text: str,
        applicable_customer_types: list[str],
        rhythm_note: str,
        boundary_note: str,
        caution_note: str,
    ) -> None:
        await self.prompt_repo.upsert_prompt(project_id, guidance_text, applicable_customer_types, rhythm_note, boundary_note, caution_note)


class MaterialService:
    def __init__(self, material_repo: MaterialRepository):
        self.material_repo = material_repo

    async def list_project_materials(self, business_account_id: int, project_id: int | None = None) -> list[asyncpg.Record]:
        return await self.material_repo.list_materials(business_account_id=business_account_id, project_id=project_id, material_scope="project")

    async def create_material(
        self,
        business_account_id: int,
        material_scope: str,
        material_type: str,
        title: str,
        description: str,
        content_text: str | None = None,
        file_url: str | None = None,
        project_id: int | None = None,
        semantic_tags_json: list[str] | None = None,
        keyword_tags_json: list[str] | None = None,
        scene_tags_json: list[str] | None = None,
        applicable_customer_tags_json: list[str] | None = None,
        applicable_stage_tags_json: list[str] | None = None,
    ) -> None:
        await self.material_repo.create_material(
            business_account_id=business_account_id,
            project_id=project_id,
            material_scope=material_scope,
            material_type=material_type,
            title=title,
            description=description,
            content_text=content_text,
            file_url=file_url,
            semantic_tags_json=semantic_tags_json or [],
            keyword_tags_json=keyword_tags_json or [],
            scene_tags_json=scene_tags_json or [],
            applicable_customer_tags_json=applicable_customer_tags_json or [],
            applicable_stage_tags_json=applicable_stage_tags_json or [],
        )


class PersonaProfileService:
    def __init__(self, profile_repo: PersonaProfileRepository, material_repo: MaterialRepository):
        self.profile_repo = profile_repo
        self.material_repo = material_repo

    async def get_active_profile(self, business_account_id: int) -> asyncpg.Record | None:
        return await self.profile_repo.get_active_profile(business_account_id)

    async def list_persona_materials(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.material_repo.list_materials(business_account_id=business_account_id, material_scope="persona")

    async def upsert_profile(self, business_account_id: int, **kwargs: Any) -> None:
        await self.profile_repo.upsert_active_profile(business_account_id=business_account_id, **kwargs)


class DailyMaterialService:
    def __init__(self, material_repo: MaterialRepository):
        self.material_repo = material_repo

    async def list_daily_materials(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.material_repo.list_materials(business_account_id=business_account_id, material_scope="daily")

    async def create_daily_material(self, business_account_id: int, material_type: str, title: str, description: str = "", content_text: str = "", file_url: str = "") -> None:
        await self.material_repo.create_material(
            business_account_id=business_account_id,
            project_id=None,
            material_scope="daily",
            material_type=material_type,
            title=title,
            description=description,
            content_text=content_text,
            file_url=file_url,
        )


class HandoverService:
    def __init__(self, handover_repo: HandoverRepository, user_repo: UserRepository):
        self.handover_repo = handover_repo
        self.user_repo = user_repo

    async def start(self, business_account_id: int, user_id: int) -> None:
        await self.handover_repo.start_handover(business_account_id, user_id)
        await self.user_repo.set_handover_state(user_id, True)

    async def end(self, business_account_id: int, user_id: int) -> None:
        await self.handover_repo.end_handover(business_account_id, user_id)
        await self.user_repo.set_handover_state(user_id, False)

    async def list_active(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.handover_repo.list_active(business_account_id)


class ReceiptService:
    def __init__(self, receipt_repo: ReceiptRepository):
        self.receipt_repo = receipt_repo

    async def list_pending(self, business_account_id: int) -> list[asyncpg.Record]:
        return await self.receipt_repo.list_pending(business_account_id)


class QueueService:
    def __init__(self, queue_repo: QueueRepository):
        self.queue_repo = queue_repo

    async def list_by_type(self, business_account_id: int, queue_type: str) -> list[asyncpg.Record]:
        return await self.queue_repo.list_by_type(business_account_id, queue_type)

    async def list_all(self, business_account_id: int) -> list[asyncpg.Record]:
        items: list[asyncpg.Record] = []
        for qt in ["high_intent", "urgent", "resume_pending", "followup"]:
            items.extend(await self.queue_repo.list_by_type(business_account_id, qt))
        return items


class AuditService:
    def __init__(self, audit_repo: AuditRepository):
        self.audit_repo = audit_repo

    async def create_log(self, business_account_id: int, user_id: int | None, audit_type: str, payload: dict[str, Any]) -> None:
        await self.audit_repo.create_audit_log(business_account_id, user_id, audit_type, payload)


# =========================================================
# Admin menu tree & callback router skeleton
# =========================================================

ADM_MAIN = "adm:main"




class UserUnderstandingEngine:
    """Extracts user intent, emotion, boundary signals, and topic tags."""

    QUESTION_WORDS = ("what", "how", "why", "when", "where", "which", "can", "could", "do", "does", "is", "are")
    BUSY_SIGNALS = ("busy", "later", "working", "sleep", "sleeping", "tomorrow", "after", "not now")
    BOUNDARY_SIGNALS = ("stop", "don't push", "too much", "later", "not now", "busy")
    POSITIVE_SIGNALS = ("interested", "sounds good", "okay", "tell me more", "curious", "profit", "return", "收益", "利润")
    HESITATION_SIGNALS = ("not sure", "maybe", "hmm", "consider", "think about", "担心", "犹豫")

    def understand(self, user_message_text: str, recent_messages: list[dict[str, Any]], user_state_snapshot: UserStateSnapshot) -> UnderstandingResult:
        text = (user_message_text or "").strip()
        lowered = text.lower()
        explicit_intent = "question" if any(lowered.startswith(w + " ") or lowered == w for w in self.QUESTION_WORDS) or "?" in text else "conversation"
        hidden_intent = "project_interest" if any(sig in lowered for sig in self.POSITIVE_SIGNALS) else "rapport"
        emotion_state = "positive" if any(sig in lowered for sig in self.POSITIVE_SIGNALS) else ("hesitant" if any(sig in lowered for sig in self.HESITATION_SIGNALS) else "neutral")
        boundary_signal = "strong" if any(sig in lowered for sig in self.BOUNDARY_SIGNALS) else "none"
        busy_signal = any(sig in lowered for sig in self.BUSY_SIGNALS)
        interest_signal = "high" if any(sig in lowered for sig in self.POSITIVE_SIGNALS) else "normal"
        hesitation_level = "high" if any(sig in lowered for sig in self.HESITATION_SIGNALS) else "low"
        readiness_level = "low" if busy_signal or boundary_signal == "strong" else ("medium" if explicit_intent == "question" else "normal")
        style_signal = "logic_first" if any(k in lowered for k in ("how", "risk", "return", "profit", "process", "收益", "流程")) else "balanced"
        topic_tags = []
        tag_map = {
            "profit": ("profit", "收益", "利润", "return", "yield"),
            "risk": ("risk", "风险", "safe", "安全"),
            "process": ("process", "流程", "how", "怎么", "步骤"),
            "persona": ("you", "your", "工作", "hobby", "兴趣", "生活"),
            "daily": ("today", "weekend", "天气", "下班", "吃饭", "爬山"),
        }
        for tag, words in tag_map.items():
            if any(w in lowered for w in words):
                topic_tags.append(tag)
        if not topic_tags:
            topic_tags.append("general")
        summary = text[:160]
        return UnderstandingResult(
            explicit_intent=explicit_intent,
            hidden_intent=hidden_intent,
            emotion_state=emotion_state,
            boundary_signal=boundary_signal,
            busy_signal=busy_signal,
            interest_signal=interest_signal,
            hesitation_level=hesitation_level,
            readiness_level=readiness_level,
            style_signal=style_signal,
            topic_tags=topic_tags,
            summary=summary,
        )


class ConversationStageEngine:
    def decide(self, understanding: UnderstandingResult, user_state_snapshot: UserStateSnapshot, recent_messages: list[dict[str, Any]]) -> StageDecision:
        prior = user_state_snapshot.last_stage or "initial_contact"
        if user_state_snapshot.intent_level in {"成交", "converted", "customer"}:
            stage = "post_conversion"
        elif understanding.interest_signal == "high" and prior in {"trust_building", "project_transition", "interest_deepening"}:
            stage = "interest_deepening"
        elif understanding.hidden_intent == "project_interest":
            stage = "project_transition"
        elif len(recent_messages) >= 6:
            stage = "trust_building"
        else:
            stage = prior or "initial_contact"
        should_pause = understanding.busy_signal or understanding.boundary_signal == "strong"
        should_hold = stage in {"initial_contact", "trust_building"}
        should_progress = stage in {"project_transition", "interest_deepening", "post_conversion"} and not should_pause
        trust_level = "high" if user_state_snapshot.trust_score >= 0.75 else ("medium" if user_state_snapshot.trust_score >= 0.4 else "low")
        engagement_level = "high" if len(recent_messages) >= 4 else "normal"
        confidence = 0.8 if stage != prior else 0.65
        return StageDecision(
            stage_key=stage,
            stage_confidence=confidence,
            trust_level=trust_level,
            engagement_level=engagement_level,
            should_hold=should_hold,
            should_progress=should_progress,
            should_pause=should_pause,
            reason=f"prior={prior}; intent={understanding.hidden_intent}; busy={understanding.busy_signal}",
        )


class RelationshipTrajectoryEngine:
    def decide(self, understanding: UnderstandingResult, stage: StageDecision, user_state_snapshot: UserStateSnapshot) -> tuple[str, str]:
        if stage.should_pause:
            return "pause_respect", "boundary or busy signal detected"
        if understanding.interest_signal == "high":
            return "warming_up", "clear interest signal"
        if user_state_snapshot.last_trajectory:
            return user_state_snapshot.last_trajectory, "keep continuity"
        return "steady", "default steady trajectory"


class RelationshipRhythmEngine:
    def decide(self, understanding: UnderstandingResult, stage: StageDecision, trajectory_key: str, recent_messages: list[dict[str, Any]]) -> tuple[str, str]:
        if stage.should_pause:
            return "low_pressure", "respect pause"
        if trajectory_key == "warming_up" and len(recent_messages) >= 4:
            return "engaged", "conversation active"
        if understanding.style_signal == "logic_first":
            return "measured", "logic-first user"
        return "steady", "default rhythm"


class RelationshipMaturityEngine:
    def decide(self, stage: StageDecision, trajectory_key: str, rhythm_key: str, user_state_snapshot: UserStateSnapshot) -> tuple[str, str]:
        if stage.stage_key == "post_conversion":
            return "high", "converted user"
        if stage.stage_key == "interest_deepening" and user_state_snapshot.trust_score >= 0.6:
            return "medium_high", "deepening with trust"
        if stage.stage_key in {"trust_building", "project_transition"}:
            return "medium", "relationship established"
        return "low", "early stage"


class RhythmConflictResolver:
    def resolve(self, understanding: UnderstandingResult, stage: StageDecision, trajectory_key: str, rhythm_key: str, maturity_key: str) -> tuple[str, list[str]]:
        risk_flags: list[str] = []
        mode = "normal"
        if understanding.boundary_signal == "strong":
            mode = "boundary_respect"
            risk_flags.append("boundary")
        elif understanding.busy_signal:
            mode = "low_interrupt"
            risk_flags.append("busy")
        elif maturity_key in {"low"} and trajectory_key == "warming_up":
            mode = "gentle_progress"
        return mode, risk_flags


class MemorySelector:
    def __init__(self, memory_repo: MemoryRepository) -> None:
        self.memory_repo = memory_repo

    async def select(self, user_id: int, understanding: UnderstandingResult, stage: StageDecision, recent_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        memories = await self.memory_repo.list_memories(user_id=user_id, limit=12)
        preferred = [m for m in memories if any(tag in (m.get("memory_key") or "") or tag in (m.get("memory_value") or "") for tag in understanding.topic_tags)]
        return preferred[:5] if preferred else memories[:5]


class MemoryPriorityResolver:
    def order(self, selected_memories: list[dict[str, Any]], understanding: UnderstandingResult, stage: StageDecision) -> list[dict[str, Any]]:
        def score(m: dict[str, Any]) -> tuple[int, int]:
            importance = int(m.get("importance") or 0)
            topical = 1 if any(tag in str(m.get("memory_key", "")) or tag in str(m.get("memory_value", "")) for tag in understanding.topic_tags) else 0
            return (topical, importance)
        return sorted(selected_memories, key=score, reverse=True)


class ContinuityGuard:
    def analyze(self, ordered_memories: list[dict[str, Any]], recent_messages: list[dict[str, Any]], user_state_snapshot: UserStateSnapshot) -> dict[str, Any]:
        recent_text = " ".join(str(m.get("content") or "") for m in recent_messages[-4:]).lower()
        suppress_keys = []
        for memory in ordered_memories:
            key = str(memory.get("memory_key") or "")
            if key and key.lower() in recent_text:
                suppress_keys.append(key)
        return {"suppress_memory_keys": suppress_keys, "continuity_hint": user_state_snapshot.recent_summary[:120] if user_state_snapshot.recent_summary else ""}


class AdaptiveStrategyOptimizer:
    def optimize(self, understanding: UnderstandingResult, stage: StageDecision, trajectory_key: str, rhythm_key: str, maturity_key: str, user_state_snapshot: UserStateSnapshot) -> tuple[dict[str, Any], dict[str, Any]]:
        profile = {
            "style": understanding.style_signal,
            "risk_sensitivity": "high" if "risk" in understanding.topic_tags else "normal",
            "space_preferring": understanding.busy_signal or understanding.boundary_signal == "strong",
        }
        bias = {
            "max_marketing_intensity": 1 if profile["space_preferring"] else (2 if maturity_key in {"low", "medium"} else 3),
            "prefer_logic": understanding.style_signal == "logic_first",
        }
        return profile, bias


class ProjectWindowEvaluator:
    def evaluate(self, turn_decision: TurnDecision, stage: StageDecision, understanding: UnderstandingResult, user_state_snapshot: UserStateSnapshot) -> str:
        if stage.should_pause:
            return "closed"
        if understanding.hidden_intent == "project_interest" or "profit" in understanding.topic_tags:
            return "open_light"
        if stage.stage_key in {"interest_deepening", "post_conversion"}:
            return "open"
        return "background_only"


class ProjectNurturePlanner:
    def plan(self, turn_decision: TurnDecision, project_window_state: str, understanding: UnderstandingResult, stage: StageDecision, user_style_profile: dict[str, Any], user_state_snapshot: UserStateSnapshot) -> dict[str, Any]:
        if project_window_state == "closed":
            return {"mode": "no_project", "note": "respect boundary"}
        if project_window_state == "background_only":
            return {"mode": "soft_presence", "note": "keep project in the background"}
        if understanding.explicit_intent == "question":
            return {"mode": "answer_question", "note": "respond to question with project context"}
        return {"mode": "light_nurture", "note": "low-pressure project nurturing"}


class TurnDecisionEngine:
    def decide(self, understanding: UnderstandingResult, stage: StageDecision, trajectory_key: str, rhythm_key: str, maturity_key: str, resolved_interaction_mode: str, ordered_memories: list[dict[str, Any]], user_style_profile: dict[str, Any], strategy_bias: dict[str, Any], user_state_snapshot: UserStateSnapshot) -> TurnDecision:
        if stage.should_pause:
            goal = "respect_boundary"
            marketing_intensity = 0
        elif understanding.explicit_intent == "question":
            goal = "answer_and_clarify"
            marketing_intensity = min(1, strategy_bias["max_marketing_intensity"])
        elif understanding.hidden_intent == "project_interest":
            goal = "nurture_interest"
            marketing_intensity = min(2, strategy_bias["max_marketing_intensity"])
        else:
            goal = "rapport_maintenance"
            marketing_intensity = 0 if user_style_profile.get("space_preferring") else 1
        should_send_media = "profit" in understanding.topic_tags and marketing_intensity >= 1
        preferred_media = ["image", "video"] if should_send_media else []
        return TurnDecision(
            turn_goal=goal,
            relationship_mode=resolved_interaction_mode,
            project_mode="project_enabled" if marketing_intensity > 0 else "no_push",
            initiative_level="low" if stage.should_pause else ("medium" if marketing_intensity <= 1 else "medium_high"),
            question_rate="low" if understanding.explicit_intent == "question" else "medium",
            reply_length="medium",
            marketing_intensity=marketing_intensity,
            should_send_media=should_send_media,
            preferred_media_types=preferred_media,
            should_follow_up_later=stage.should_pause,
            should_end_softly=stage.should_pause,
            reason=f"goal={goal}; style={user_style_profile.get('style')}"
        )


class ProjectPromptResolver:
    def __init__(self, project_prompt_repo: ProjectPromptRepository) -> None:
        self.project_prompt_repo = project_prompt_repo

    async def resolve(self, current_project_id: int | None, project_window_state: str, user_style_profile: dict[str, Any]) -> str:
        if not current_project_id:
            return ""
        row = await self.project_prompt_repo.get_prompt(current_project_id)
        return (row.get("guidance_text") if row else "") or ""


class ProjectFAQResolver:
    def __init__(self, project_faq_repo: ProjectFAQRepository) -> None:
        self.project_faq_repo = project_faq_repo

    async def resolve(self, current_project_id: int | None, understanding: UnderstandingResult) -> list[FAQMatchResult]:
        if not current_project_id:
            return []
        rows = await self.project_faq_repo.list_by_project(current_project_id)
        results = []
        for row in rows:
            question = str(row.get("question") or "")
            answer = str(row.get("answer") or "")
            score = 0.0
            for tag in understanding.topic_tags:
                if tag in question.lower() or tag in answer.lower():
                    score += 1.0
            if score > 0:
                results.append(FAQMatchResult(faq_id=row["id"], question=question, answer=answer, category=row.get("category") or "general", score=score, reason="topic tag matched"))
        return sorted(results, key=lambda x: x.score, reverse=True)[:3]


class MarketingPlanResolver:
    STAGE_MAP = {
        "nurture_interest": "interested_customer",
        "answer_and_clarify": "first_product_intro",
        "rapport_maintenance": "first_product_intro",
    }

    def __init__(self, marketing_plan_repo: MarketingPlanRepository, marketing_plan_segment_repo: MarketingPlanSegmentRepository) -> None:
        self.marketing_plan_repo = marketing_plan_repo
        self.marketing_plan_segment_repo = marketing_plan_segment_repo

    async def resolve(self, current_project_id: int | None, turn_decision: TurnDecision, user_style_profile: dict[str, Any], stage: StageDecision, understanding: UnderstandingResult) -> list[MarketingPlanMatchResult]:
        if not current_project_id or turn_decision.marketing_intensity == 0:
            return []
        stage_type = self.STAGE_MAP.get(turn_decision.turn_goal, "first_product_intro")
        plans = await self.marketing_plan_repo.list_by_stage_type(current_project_id, stage_type)
        matches: list[MarketingPlanMatchResult] = []
        for plan in plans:
            segments = await self.marketing_plan_segment_repo.list_by_plan(plan["id"])
            segment_payload = [
                {"segment_id": seg["id"], "title": seg.get("segment_title") or "", "content": seg.get("content") or "", "segment_type": seg.get("segment_type") or "general"}
                for seg in segments if seg.get("is_enabled", True)
            ]
            matches.append(MarketingPlanMatchResult(
                plan_id=plan["id"],
                customer_stage_type=stage_type,
                intensity_level=int(plan.get("intensity_level") or 0),
                matched_customer_tags=list(plan.get("customer_tags_json") or []),
                matched_segments=segment_payload[:4],
                reason="stage and intensity matched",
            ))
        return matches[:2]


class PersonaProfileResolver:
    def __init__(self, persona_profile_repo: PersonaProfileRepository) -> None:
        self.persona_profile_repo = persona_profile_repo

    async def resolve(self, business_account_id: int, understanding: UnderstandingResult) -> tuple[PersonaProfileSnapshot | None, dict[str, str]]:
        row = await self.persona_profile_repo.get_active_profile(business_account_id)
        if not row:
            return None, {}
        snapshot = PersonaProfileSnapshot(
            name=row.get("name") or "",
            gender=row.get("gender") or "female",
            age=str(row.get("age") or ""),
            height=str(row.get("height") or ""),
            weight=str(row.get("weight") or ""),
            nationality=row.get("nationality") or "",
            current_residence=row.get("current_residence") or "",
            marital_status=row.get("marital_status") or "",
            has_children=str(row.get("has_children") or ""),
            employer=row.get("employer") or "",
            job_title=row.get("job_title") or "",
            hobbies=row.get("hobbies") or "",
            financial_status=row.get("financial_status") or "",
            relationship_history=row.get("relationship_history") or "",
        )
        relevant_fields = {}
        if "persona" in understanding.topic_tags:
            relevant_fields = {
                "job_title": snapshot.job_title,
                "hobbies": snapshot.hobbies,
                "current_residence": snapshot.current_residence,
            }
        return snapshot, relevant_fields


class MaterialRetriever:
    def __init__(self, material_repo: MaterialRepository) -> None:
        self.material_repo = material_repo

    async def retrieve(self, business_account_id: int, current_project_id: int | None, topic_tags: list[str], scopes: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for scope in scopes:
            rows = await self.material_repo.list_materials(
                business_account_id=business_account_id,
                project_id=current_project_id if scope == "project" else None,
                material_scope=scope,
            )
            enabled_rows = [row for row in rows if row.get("is_enabled", True)]
            results.extend(enabled_rows)
        return results


class MaterialRelevanceRanker:
    def rank(self, materials: list[dict[str, Any]], topic_tags: list[str]) -> list[MaterialMatchResult]:
        ranked: list[MaterialMatchResult] = []
        for row in materials:
            haystack = " ".join(
                [
                    str(row.get("title") or ""),
                    str(row.get("description") or ""),
                    str(row.get("content_text") or ""),
                    " ".join(row.get("semantic_tags_json") or []),
                    " ".join(row.get("keyword_tags_json") or []),
                    " ".join(row.get("scene_tags_json") or []),
                ]
            ).lower()
            semantic_score = sum(1.0 for tag in topic_tags if tag in haystack)
            scene_score = 1.0 if any(tag in (row.get("scene_tags_json") or []) for tag in topic_tags) else 0.0
            final_score = semantic_score + scene_score + float(row.get("priority") or 0) * 0.01
            if final_score > 0:
                ranked.append(MaterialMatchResult(
                    material_id=row["id"],
                    material_scope=row.get("material_scope") or "project",
                    material_type=row.get("material_type") or "text",
                    title=row.get("title") or "",
                    content_text=row.get("content_text"),
                    file_url=row.get("file_url"),
                    semantic_score=semantic_score,
                    scene_score=scene_score,
                    final_score=final_score,
                    reason="semantic topic match",
                ))
        return sorted(ranked, key=lambda x: x.final_score, reverse=True)


class MaterialSelectionPlanner:
    def select(self, ranked_materials: list[MaterialMatchResult], turn_decision: TurnDecision, stage: StageDecision) -> list[MaterialMatchResult]:
        if not turn_decision.should_send_media:
            text_only = [m for m in ranked_materials if m.material_type == "text"]
            return text_only[:2]
        selected: list[MaterialMatchResult] = []
        used_types: set[str] = set()
        for material in ranked_materials:
            if material.material_type in turn_decision.preferred_media_types or material.material_type == "text":
                if material.material_type not in used_types or material.material_type == "text":
                    selected.append(material)
                    used_types.add(material.material_type)
            if len(selected) >= 3:
                break
        return selected


class HumanizationController:
    def build_style_and_reply(self, turn_decision: TurnDecision, project_nurture_plan: dict[str, Any], faq_matches: list[FAQMatchResult], marketing_plan_matches: list[MarketingPlanMatchResult], selected_materials: list[MaterialMatchResult], persona_relevant_fields: dict[str, str], ordered_memories: list[dict[str, Any]], project_prompt_guidance: str, user_style_profile: dict[str, Any], understanding: UnderstandingResult) -> tuple[StyleSpec, str]:
        warmth = "warm" if turn_decision.marketing_intensity <= 1 else "warm_business"
        formality = "medium" if user_style_profile.get("style") == "logic_first" else "casual_medium"
        style = StyleSpec(
            warmth_level=warmth,
            formality_level=formality,
            distance_level="respectful_close",
            initiative_tone=turn_decision.initiative_level,
            question_style=turn_decision.question_rate,
            business_tone="light_business",
            life_tone="soft_life" if "daily" in understanding.topic_tags else "neutral",
            maturity_tone="steady",
            emoji_level="low",
            cadence_note=project_prompt_guidance[:120] if project_prompt_guidance else "",
        )
        parts: list[str] = []
        if understanding.emotion_state == "hesitant":
            parts.append("I get why you'd want to look at this carefully.")
        if faq_matches:
            parts.append(faq_matches[0].answer)
        elif marketing_plan_matches and marketing_plan_matches[0].matched_segments:
            parts.append(marketing_plan_matches[0].matched_segments[0]["content"])
        elif project_nurture_plan.get("mode") == "soft_presence":
            parts.append("I don't want to make this feel pushy. I can just give you the key point first.")
        else:
            parts.append("I can explain it in a simple way based on what matters most to you.")
        if persona_relevant_fields.get("hobbies") and "persona" in understanding.topic_tags:
            parts.append(f"By the way, outside work I’m also into {persona_relevant_fields['hobbies']}, so I usually prefer keeping things practical and easy to follow.")
        if selected_materials:
            parts.append("I’ll also share the most relevant reference so it’s easier to see.")
        draft = " ".join(p.strip() for p in parts if p).strip()
        return style, draft


class ReplySelfCheckEngine:
    def check(self, draft_reply_text: str, turn_decision: TurnDecision, risk_flags: list[str], project_prompt_guidance: str) -> tuple[str, list[str]]:
        notes: list[str] = []
        text = draft_reply_text.strip()
        if turn_decision.marketing_intensity == 0 and "buy" in text.lower():
            text = text.replace("buy", "look at")
            notes.append("reduced sales pressure")
        if "boundary" in risk_flags:
            text = text + " No rush at all."
            notes.append("added boundary-safe close")
        if len(text) > 900:
            text = text[:900].rsplit(" ", 1)[0] + "…"
            notes.append("trimmed length")
        return text, notes


class MaturityPolishEngine:
    def polish(self, checked_reply_text: str, style_spec: StyleSpec, maturity_key: str) -> str:
        text = re.sub(r"\s{2,}", " ", checked_reply_text).strip()
        if maturity_key in {"medium_high", "high"} and not text.endswith("."):
            text += "."
        return text


class MemoryWritebackEngine:
    def __init__(self, memory_repo: MemoryRepository, conversation_repo: ConversationRepository) -> None:
        self.memory_repo = memory_repo
        self.conversation_repo = conversation_repo

    async def write_after_turn(self, business_account_id: int, user_id: int, understanding: UnderstandingResult, stage: StageDecision, trajectory_key: str, rhythm_key: str, maturity_key: str, final_reply_text: str) -> None:
        memory_key = understanding.topic_tags[0] if understanding.topic_tags else "general"
        await self.memory_repo.create_memory(
            business_account_id=business_account_id,
            user_id=user_id,
            memory_type="turn_summary",
            memory_key=memory_key,
            memory_value=understanding.summary[:240],
            importance=50,
            source="ai",
        )
        await self.conversation_repo.upsert_state(
            business_account_id=business_account_id,
            user_id=user_id,
            last_stage=stage.stage_key,
            last_trajectory=trajectory_key,
            last_rhythm=rhythm_key,
            last_maturity=maturity_key,
            trust_score=0.6 if stage.trust_level == "medium" else (0.85 if stage.trust_level == "high" else 0.3),
            summary_text=final_reply_text[:240],
        )


class ManualStrategyAbstractor:
    def abstract(self, handover_summary: str) -> dict[str, Any]:
        return {"summary": handover_summary[:240], "tone": "measured", "followup_bias": "gentle"}


class HandoverLearningEngine:
    def __init__(self, handover_repo: HandoverRepository, audit_repo: AuditRepository) -> None:
        self.handover_repo = handover_repo
        self.audit_repo = audit_repo
        self.abstractor = ManualStrategyAbstractor()

    async def learn(self, business_account_id: int, user_id: int, final_reply_text: str) -> None:
        active_items = await self.handover_repo.list_active(business_account_id)
        active = next((row for row in active_items if row.get("user_id") == user_id), None)
        if not active:
            return
        summary = active.get("handover_summary") or ""
        abstract = self.abstractor.abstract(summary)
        await self.audit_repo.create_audit_log(
            business_account_id=business_account_id,
            user_id=user_id,
            audit_type="handover_learning",
            payload={"abstract": abstract, "reply_excerpt": final_reply_text[:200]},
        )


class ResumeBridgePlanner:
    def build(self, latest_handover_summary: str, stage: StageDecision, user_state_snapshot: UserStateSnapshot) -> str:
        if not latest_handover_summary:
            return ""
        return f"Resume gently from the latest handover context while keeping the user in {stage.stage_key} stage."

def build_admin_main_menu() -> list[list[dict[str, str]]]:
    return [
        [{"text": "仪表盘", "callback_data": "adm:dashboard:view"}],
        [{"text": "用户管理", "callback_data": "adm:users:list"}],
        [{"text": "用户分类", "callback_data": "adm:classification:view"}],
        [{"text": "项目管理", "callback_data": "adm:projects:list"}],
        [{"text": "素材管理", "callback_data": "adm:materials:project"}],
        [{"text": "人设素材", "callback_data": "adm:persona:profile:view"}],
        [{"text": "接管管理", "callback_data": "adm:handover:list_active"}],
        [{"text": "队列中心", "callback_data": "adm:queues:all"}],
        [{"text": "回执审核", "callback_data": "adm:receipts:pending"}],
        [{"text": "系统设置", "callback_data": "adm:settings:system_status"}],
    ]


def build_simple_menu(title: str, items: list[tuple[str, str]], back_callback: str = ADM_MAIN) -> dict[str, Any]:
    return {
        "title": title,
        "items": [{"text": text, "callback": cb} for text, cb in items],
        "back": back_callback,
    }


class TGAdminCallbackRouter:
    MARKETING_STAGE_MAP = {
        "first_intro": "first_product_intro",
        "interested": "interested_customer",
        "closing": "closing_customer",
        "converted": "converted_customer",
        "upgrade": "upgrade_customer",
    }

    def __init__(self, admin_api_service: AdminAPIService, dashboard_service: DashboardService):
        self.admin_api_service = admin_api_service
        self.dashboard_service = dashboard_service

    async def route(self, business_account_id: int, callback_data: str) -> dict[str, Any]:
        parts = callback_data.split(":")
        if not parts or parts[0] != "adm":
            return {"ok": False, "message": "无效的管理员回调。"}
        if callback_data == ADM_MAIN:
            return {"ok": True, "view": "main", "menu": build_admin_main_menu()}
        if len(parts) < 3:
            return {"ok": False, "message": "管理员回调格式错误。"}
        module, action = parts[1], parts[2]

        if module == "dashboard":
            summary = await self.dashboard_service.build_dashboard_summary(business_account_id)
            return {"ok": True, "view": "dashboard", "data": summary}

        if module == "users":
            if action == "list":
                users = await self.admin_api_service.get_user_list(business_account_id)
                return {"ok": True, "view": "users:list", "data": users}
            if action == "view" and len(parts) >= 4:
                user_id = int(parts[3])
                detail = await self.admin_api_service.get_user_detail(business_account_id, user_id)
                return {"ok": True, "view": "users:view", "data": detail}
            if action == "handover_start" and len(parts) >= 4:
                user_id = int(parts[3])
                await self.admin_api_service.start_handover(business_account_id, user_id)
                return {"ok": True, "view": "users:handover_start", "message": f"已开始接管用户 {user_id}。"}
            if action == "resume_ai" and len(parts) >= 4:
                user_id = int(parts[3])
                await self.admin_api_service.end_handover(business_account_id, user_id)
                return {"ok": True, "view": "users:resume_ai", "message": f"已恢复用户 {user_id} 的 AI。"}

        if module == "projects":
            if action == "list":
                projects = await self.admin_api_service.get_projects(business_account_id)
                return {"ok": True, "view": "projects:list", "data": projects}
            if action == "view" and len(parts) >= 4:
                project_id = int(parts[3])
                detail = await self.admin_api_service.get_project_detail(project_id)
                return {"ok": True, "view": "projects:view", "data": detail}

        if module == "faq":
            if action == "list" and len(parts) >= 4:
                project_id = int(parts[3])
                faqs = await self.admin_api_service.get_project_faqs(project_id)
                return {"ok": True, "view": "faq:list", "data": faqs, "project_id": project_id}

        if module == "marketing":
            if action in self.MARKETING_STAGE_MAP and len(parts) >= 4:
                project_id = int(parts[3])
                plans = await self.admin_api_service.get_marketing_plans(project_id, self.MARKETING_STAGE_MAP[action])
                return {"ok": True, "view": f"marketing:{action}", "data": plans, "project_id": project_id}
            if action == "view" and len(parts) >= 4:
                plan_id = int(parts[3])
                segments = await self.admin_api_service.get_marketing_segments(plan_id)
                return {"ok": True, "view": "marketing:view", "data": segments, "plan_id": plan_id}

        if module == "marketing_segments":
            if action == "list" and len(parts) >= 4:
                plan_id = int(parts[3])
                segments = await self.admin_api_service.get_marketing_segments(plan_id)
                return {"ok": True, "view": "marketing_segments:list", "data": segments, "plan_id": plan_id}

        if module == "project_prompts":
            if action == "view" and len(parts) >= 4:
                project_id = int(parts[3])
                prompt = await self.admin_api_service.get_project_prompt(project_id)
                return {"ok": True, "view": "project_prompts:view", "data": prompt, "project_id": project_id}

        if module == "materials":
            if action == "project" and len(parts) >= 4 and parts[3] == "list":
                materials = await self.admin_api_service.get_materials(business_account_id, scope="project")
                return {"ok": True, "view": "materials:project:list", "data": materials}
            if action == "project" and len(parts) >= 5 and parts[3] == "by_project":
                project_id = int(parts[4])
                materials = await self.admin_api_service.get_materials(business_account_id, scope="project", project_id=project_id)
                return {"ok": True, "view": "materials:project:by_project", "data": materials, "project_id": project_id}
            if action == "persona" and len(parts) >= 4 and parts[3] == "list":
                materials = await self.admin_api_service.get_persona_materials(business_account_id)
                return {"ok": True, "view": "materials:persona:list", "data": materials}
            if action == "daily" and len(parts) >= 4 and parts[3] == "list":
                materials = await self.admin_api_service.get_daily_materials(business_account_id)
                return {"ok": True, "view": "materials:daily:list", "data": materials}

        if module == "persona":
            if action == "profile" and len(parts) >= 4 and parts[3] == "view":
                profile = await self.admin_api_service.get_persona_profile(business_account_id)
                return {"ok": True, "view": "persona:profile:view", "data": profile}
            if action == "materials" and len(parts) >= 4 and parts[3] == "list":
                materials = await self.admin_api_service.get_persona_materials(business_account_id)
                return {"ok": True, "view": "persona:materials:list", "data": materials}

        if module == "handover":
            if action == "list_active":
                items = await self.admin_api_service.get_active_handovers(business_account_id)
                return {"ok": True, "view": "handover:list_active", "data": items}
            if action == "start" and len(parts) >= 4:
                user_id = int(parts[3])
                await self.admin_api_service.start_handover(business_account_id, user_id)
                return {"ok": True, "view": "handover:start", "message": f"已开始接管用户 {user_id}。"}
            if action == "end" and len(parts) >= 4:
                user_id = int(parts[3])
                await self.admin_api_service.end_handover(business_account_id, user_id)
                return {"ok": True, "view": "handover:end", "message": f"已结束接管用户 {user_id}。"}

        if module == "receipts":
            if action == "pending":
                receipts = await self.admin_api_service.get_pending_receipts(business_account_id)
                return {"ok": True, "view": "receipts:pending", "data": receipts}

        if module == "queues":
            queue_type = action
            items = await self.admin_api_service.get_queue_items(business_account_id, queue_type)
            return {"ok": True, "view": f"queues:{queue_type}", "data": items}

        return {"ok": True, "view": "placeholder", "callback_data": callback_data}


@dataclass(slots=True)
class OutboundReplyPayload:
    business_account_id: int
    user_id: int
    reply_text: str
    style_spec: StyleSpec
    selected_materials: list[MaterialMatchResult] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    def __init__(
        self,
        user_repo: UserRepository,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        memory_repo: MemoryRepository,
        handover_repo: HandoverRepository,
        understanding_engine: UserUnderstandingEngine,
        stage_engine: ConversationStageEngine,
        trajectory_engine: RelationshipTrajectoryEngine,
        rhythm_engine: RelationshipRhythmEngine,
        maturity_engine: RelationshipMaturityEngine,
        conflict_resolver: RhythmConflictResolver,
        memory_selector: MemorySelector,
        memory_priority_resolver: MemoryPriorityResolver,
        continuity_guard: ContinuityGuard,
        strategy_optimizer: AdaptiveStrategyOptimizer,
        turn_decision_engine: TurnDecisionEngine,
        project_window_evaluator: ProjectWindowEvaluator,
        project_nurture_planner: ProjectNurturePlanner,
        project_prompt_resolver: ProjectPromptResolver,
        project_faq_resolver: ProjectFAQResolver,
        marketing_plan_resolver: MarketingPlanResolver,
        persona_profile_resolver: PersonaProfileResolver,
        material_retriever: MaterialRetriever,
        material_ranker: MaterialRelevanceRanker,
        material_selection_planner: MaterialSelectionPlanner,
        humanization_controller: HumanizationController,
        reply_selfcheck_engine: ReplySelfCheckEngine,
        maturity_polish_engine: MaturityPolishEngine,
        memory_writeback_engine: MemoryWritebackEngine,
        handover_learning_engine: HandoverLearningEngine,
        resume_bridge_planner: ResumeBridgePlanner,
    ) -> None:
        self.user_repo = user_repo
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.memory_repo = memory_repo
        self.handover_repo = handover_repo
        self.understanding_engine = understanding_engine
        self.stage_engine = stage_engine
        self.trajectory_engine = trajectory_engine
        self.rhythm_engine = rhythm_engine
        self.maturity_engine = maturity_engine
        self.conflict_resolver = conflict_resolver
        self.memory_selector = memory_selector
        self.memory_priority_resolver = memory_priority_resolver
        self.continuity_guard = continuity_guard
        self.strategy_optimizer = strategy_optimizer
        self.turn_decision_engine = turn_decision_engine
        self.project_window_evaluator = project_window_evaluator
        self.project_nurture_planner = project_nurture_planner
        self.project_prompt_resolver = project_prompt_resolver
        self.project_faq_resolver = project_faq_resolver
        self.marketing_plan_resolver = marketing_plan_resolver
        self.persona_profile_resolver = persona_profile_resolver
        self.material_retriever = material_retriever
        self.material_ranker = material_ranker
        self.material_selection_planner = material_selection_planner
        self.humanization_controller = humanization_controller
        self.reply_selfcheck_engine = reply_selfcheck_engine
        self.maturity_polish_engine = maturity_polish_engine
        self.memory_writeback_engine = memory_writeback_engine
        self.handover_learning_engine = handover_learning_engine
        self.resume_bridge_planner = resume_bridge_planner

    async def handle_user_message(
        self,
        business_account_id: int,
        telegram_user_id: int,
        message_text: str,
        username: str | None = None,
        display_name: str | None = None,
    ) -> OutboundReplyPayload:
        await self.user_repo.create_or_update_user(
            business_account_id=business_account_id,
            telegram_user_id=telegram_user_id,
            username=username,
            display_name=display_name,
        )
        user_row = await self.user_repo.get_by_telegram_user_id(
            business_account_id=business_account_id,
            telegram_user_id=telegram_user_id,
        )
        if not user_row:
            raise RuntimeError("User could not be loaded after upsert.")
        user_id = int(user_row["id"])

        await self.message_repo.create_message(
            business_account_id=business_account_id,
            user_id=user_id,
            role="user",
            content=message_text,
            content_type="text",
            telegram_message_id=None,
            media_url=None,
        )

        user_state_snapshot = await self.conversation_repo.get_snapshot(
            business_account_id=business_account_id,
            user_id=user_id,
        )
        recent_messages = await self.message_repo.list_recent_messages(
            business_account_id=business_account_id,
            user_id=user_id,
            limit=MAX_RECENT_MESSAGES,
        )

        understanding = self.understanding_engine.understand(
            user_message_text=message_text,
            recent_messages=recent_messages,
            user_state_snapshot=user_state_snapshot,
        )
        stage = self.stage_engine.decide(understanding, user_state_snapshot, recent_messages)
        trajectory_key, trajectory_reason = self.trajectory_engine.decide(understanding, stage, user_state_snapshot)
        rhythm_key, rhythm_reason = self.rhythm_engine.decide(understanding, stage, trajectory_key, recent_messages)
        maturity_key, maturity_reason = self.maturity_engine.decide(stage, trajectory_key, rhythm_key, user_state_snapshot)
        resolved_interaction_mode, risk_flags = self.conflict_resolver.resolve(understanding, stage, trajectory_key, rhythm_key, maturity_key)

        selected_memories = await self.memory_selector.select(user_id, understanding, stage, recent_messages)
        ordered_memories = self.memory_priority_resolver.order(selected_memories, understanding, stage)
        continuity_notes = self.continuity_guard.analyze(ordered_memories, recent_messages, user_state_snapshot)
        user_style_profile, strategy_bias = self.strategy_optimizer.optimize(
            understanding, stage, trajectory_key, rhythm_key, maturity_key, user_state_snapshot
        )

        turn_decision = self.turn_decision_engine.decide(
            understanding,
            stage,
            trajectory_key,
            rhythm_key,
            maturity_key,
            resolved_interaction_mode,
            ordered_memories,
            user_style_profile,
            strategy_bias,
            user_state_snapshot,
        )
        project_window_state = self.project_window_evaluator.evaluate(turn_decision, stage, understanding, user_state_snapshot)
        project_nurture_plan = self.project_nurture_planner.plan(
            turn_decision,
            project_window_state,
            understanding,
            stage,
            user_style_profile,
            user_state_snapshot,
        )

        current_project_id = user_state_snapshot.current_project_id
        project_prompt_guidance = await self.project_prompt_resolver.resolve(current_project_id, project_window_state, user_style_profile)
        faq_matches = await self.project_faq_resolver.resolve(current_project_id, understanding)
        marketing_plan_matches = await self.marketing_plan_resolver.resolve(
            current_project_id,
            turn_decision,
            user_style_profile,
            stage,
            understanding,
        )
        persona_profile_snapshot, persona_relevant_fields = await self.persona_profile_resolver.resolve(
            business_account_id,
            understanding,
        )

        scopes = self._select_material_scopes(understanding, current_project_id)
        raw_materials = await self.material_retriever.retrieve(
            business_account_id=business_account_id,
            current_project_id=current_project_id,
            topic_tags=understanding.topic_tags,
            scopes=scopes,
        )
        ranked_materials = self.material_ranker.rank(raw_materials, understanding.topic_tags)
        selected_materials = self.material_selection_planner.select(ranked_materials, turn_decision, stage)

        style_spec, draft_reply_text = self.humanization_controller.build_style_and_reply(
            turn_decision,
            project_nurture_plan,
            faq_matches,
            marketing_plan_matches,
            selected_materials,
            persona_relevant_fields,
            ordered_memories,
            project_prompt_guidance,
            user_style_profile,
            understanding,
        )
        checked_reply_text, selfcheck_notes = self.reply_selfcheck_engine.check(
            draft_reply_text,
            turn_decision,
            risk_flags,
            project_prompt_guidance,
        )
        final_reply_text = self.maturity_polish_engine.polish(checked_reply_text, style_spec, maturity_key)

        await self.message_repo.create_message(
            business_account_id=business_account_id,
            user_id=user_id,
            role="assistant",
            content=final_reply_text,
            content_type="text",
            telegram_message_id=None,
            media_url=None,
        )
        await self.memory_writeback_engine.write_after_turn(
            business_account_id=business_account_id,
            user_id=user_id,
            understanding=understanding,
            stage=stage,
            trajectory_key=trajectory_key,
            rhythm_key=rhythm_key,
            maturity_key=maturity_key,
            final_reply_text=final_reply_text,
        )
        await self.handover_learning_engine.learn(
            business_account_id=business_account_id,
            user_id=user_id,
            final_reply_text=final_reply_text,
        )

        return OutboundReplyPayload(
            business_account_id=business_account_id,
            user_id=user_id,
            reply_text=final_reply_text,
            style_spec=style_spec,
            selected_materials=selected_materials,
            debug={
                "understanding": understanding,
                "stage": stage,
                "trajectory": trajectory_key,
                "trajectory_reason": trajectory_reason,
                "rhythm": rhythm_key,
                "rhythm_reason": rhythm_reason,
                "maturity": maturity_key,
                "maturity_reason": maturity_reason,
                "risk_flags": risk_flags,
                "continuity_notes": continuity_notes,
                "project_window_state": project_window_state,
                "project_nurture_plan": project_nurture_plan,
                "project_prompt_guidance": project_prompt_guidance,
                "faq_match_count": len(faq_matches),
                "marketing_match_count": len(marketing_plan_matches),
                "persona_profile": persona_profile_snapshot,
                "selfcheck_notes": selfcheck_notes,
            },
        )

    def _select_material_scopes(self, understanding: UnderstandingResult, current_project_id: int | None) -> list[str]:
        scopes: list[str] = []
        if current_project_id is not None:
            scopes.append("project")
        if "persona" in understanding.topic_tags:
            scopes.append("persona")
        if "daily" in understanding.topic_tags:
            scopes.append("daily")
        if not scopes:
            scopes = ["project"] if current_project_id is not None else ["persona", "daily"]
        return scopes



# =========================================================
# Runtime integration helpers
# =========================================================

class TelegramBotClient:
    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}" if bot_token else ""

    async def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            logger.warning("BOT_TOKEN is empty; skipping Telegram API call for %s", method)
            return {"ok": False, "skipped": True, "method": method}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/{method}", json=payload)
            response.raise_for_status()
            return response.json()

    async def send_message(self, chat_id: int, text: str) -> dict[str, Any]:
        return await self._post("sendMessage", {"chat_id": chat_id, "text": text})

    async def send_photo(self, chat_id: int, photo: str, caption: str = "") -> dict[str, Any]:
        return await self._post("sendPhoto", {"chat_id": chat_id, "photo": photo, "caption": caption})

    async def send_video(self, chat_id: int, video: str, caption: str = "") -> dict[str, Any]:
        return await self._post("sendVideo", {"chat_id": chat_id, "video": video, "caption": caption})

    async def answer_callback_query(self, callback_query_id: str, text: str = "") -> dict[str, Any]:
        payload = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        return await self._post("answerCallbackQuery", payload)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str) -> dict[str, Any]:
        return await self._post("editMessageText", {"chat_id": chat_id, "message_id": message_id, "text": text})

    async def set_webhook(self, public_base_url: str, webhook_path: str, secret: str = "") -> dict[str, Any]:
        payload = {"url": f"{public_base_url.rstrip('/')}{webhook_path}"}
        if secret:
            payload["secret_token"] = secret
        return await self._post("setWebhook", payload)


class OutboundSender:
    def __init__(self, bot_client: TelegramBotClient) -> None:
        self.bot_client = bot_client

    async def send_reply(self, chat_id: int, payload: OutboundReplyPayload) -> None:
        sent_media = False
        for material in payload.selected_materials:
            if material.material_type == "image" and material.file_url:
                await self.bot_client.send_photo(chat_id, material.file_url, caption=payload.reply_text if not sent_media else "")
                sent_media = True
            elif material.material_type == "video" and material.file_url:
                await self.bot_client.send_video(chat_id, material.file_url, caption=payload.reply_text if not sent_media else "")
                sent_media = True
        if not sent_media or not payload.selected_materials:
            await self.bot_client.send_message(chat_id, payload.reply_text)


def render_admin_route_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return result.get("message", "操作失败。")
    view = result.get("view", "")
    data = result.get("data")
    if view == "main":
        return "管理员主菜单已打开。"
    if view == "dashboard" and isinstance(data, dict):
        return (
            "仪表盘\n"
            f"高意向用户：{data.get('high_intent_users', 0)}\n"
            f"接管中：{data.get('active_handovers', 0)}\n"
            f"待恢复：{data.get('pending_resumes', 0)}\n"
            f"待审核回执：{data.get('pending_receipts', 0)}\n"
            f"开放队列：{data.get('open_queues', 0)}"
        )
    if isinstance(data, list):
        if not data:
            return f"{view}：暂无数据。"
        preview = []
        for row in data[:10]:
            if isinstance(row, dict):
                preview.append(str(row.get('project_name') or row.get('display_name') or row.get('question') or row.get('reason') or row.get('id')))
            else:
                preview.append(str(dict(row).get('project_name') if hasattr(row, 'keys') else row))
        return f"{view}：\n- " + "\n- ".join(preview)
    return result.get("message", f"已进入 {view}。")


class TelegramWebhookGateway:
    def __init__(self, app_components: AppComponents, bot_client: TelegramBotClient, outbound_sender: OutboundSender):
        self.app_components = app_components
        self.bot_client = bot_client
        self.outbound_sender = outbound_sender

    async def handle_update(self, update: dict[str, Any]) -> dict[str, Any]:
        if callback := update.get("callback_query"):
            return await self._handle_callback_query(callback)
        if message := update.get("message"):
            return await self._handle_message(message)
        return {"ok": True, "ignored": True}

    async def _handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        text = compact_text(message.get("text") or "")
        if not text:
            return {"ok": True, "ignored": True, "reason": "non_text_message"}
        chat = message.get("chat") or {}
        from_user = message.get("from") or {}
        chat_id = int(chat.get("id") or from_user.get("id") or 0)
        telegram_user_id = int(from_user.get("id") or chat_id)
        username = from_user.get("username")
        display_name = compact_text(" ".join(filter(None, [from_user.get("first_name"), from_user.get("last_name")]))) or username or str(telegram_user_id)

        if text in {"/admin", "/menu"}:
            await self.bot_client.send_message(chat_id, "管理员主菜单已就绪。")
            return {"ok": True, "admin_menu": True}

        payload = await self.app_components.orchestrator.handle_user_message(
            business_account_id=DEFAULT_BUSINESS_ACCOUNT_ID,
            telegram_user_id=telegram_user_id,
            message_text=text,
            username=username,
            display_name=display_name,
        )
        await self.outbound_sender.send_reply(chat_id, payload)
        return {"ok": True, "reply_sent": True}

    async def _handle_callback_query(self, callback_query: dict[str, Any]) -> dict[str, Any]:
        callback_id = str(callback_query.get("id") or "")
        data = str(callback_query.get("data") or "")
        message = callback_query.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = int(chat.get("id") or 0)
        route_result = await self.app_components.admin_router.route(DEFAULT_BUSINESS_ACCOUNT_ID, data)
        text = render_admin_route_result(route_result)
        if callback_id:
            await self.bot_client.answer_callback_query(callback_id, text[:180])
        if chat_id and message.get("message_id"):
            try:
                await self.bot_client.edit_message_text(chat_id, int(message["message_id"]), text)
            except Exception:
                await self.bot_client.send_message(chat_id, text)
        elif chat_id:
            await self.bot_client.send_message(chat_id, text)
        return {"ok": True, "admin_result": route_result}


def create_flask_app(app_components: AppComponents, bot_client: TelegramBotClient, outbound_sender: OutboundSender) -> Flask:
    flask_app = Flask(__name__)
    gateway = TelegramWebhookGateway(app_components, bot_client, outbound_sender)

    def run_async(coro: Any) -> Any:
        return asyncio.run(coro)

    @flask_app.get('/healthz')
    def healthz() -> Any:
        return jsonify({"ok": True, "service": "tg_business_ai", "timestamp": utcnow().isoformat()})

    @flask_app.post(WEBHOOK_PATH)
    def telegram_webhook() -> Any:
        if WEBHOOK_SECRET:
            provided = request.headers.get('X-Telegram-Bot-Api-Secret-Token', '')
            if provided != WEBHOOK_SECRET:
                return jsonify({"ok": False, "error": "forbidden"}), 403
        update = request.get_json(silent=True) or {}
        result = run_async(gateway.handle_update(update))
        return jsonify(result)

    return flask_app

# =========================================================
# Composition root
# =========================================================

@dataclass(slots=True)
class AppComponents:
    db: Database
    user_repo: UserRepository
    conversation_repo: ConversationRepository
    message_repo: MessageRepository
    project_repo: ProjectRepository
    project_prompt_repo: ProjectPromptRepository
    project_faq_repo: ProjectFAQRepository
    marketing_plan_repo: MarketingPlanRepository
    marketing_plan_segment_repo: MarketingPlanSegmentRepository
    material_repo: MaterialRepository
    persona_profile_repo: PersonaProfileRepository
    memory_repo: MemoryRepository
    handover_repo: HandoverRepository
    receipt_repo: ReceiptRepository
    queue_repo: QueueRepository
    audit_repo: AuditRepository
    admin_api_service: AdminAPIService
    dashboard_service: DashboardService
    admin_router: TGAdminCallbackRouter
    orchestrator: Orchestrator


async def build_app_components() -> AppComponents:
    db = Database(DATABASE_URL)
    await db.connect()

    user_repo = UserRepository(db)
    conversation_repo = ConversationRepository(db)
    message_repo = MessageRepository(db)
    project_repo = ProjectRepository(db)
    project_prompt_repo = ProjectPromptRepository(db)
    project_faq_repo = ProjectFAQRepository(db)
    marketing_plan_repo = MarketingPlanRepository(db)
    marketing_plan_segment_repo = MarketingPlanSegmentRepository(db)
    material_repo = MaterialRepository(db)
    persona_profile_repo = PersonaProfileRepository(db)
    memory_repo = MemoryRepository(db)
    handover_repo = HandoverRepository(db)
    receipt_repo = ReceiptRepository(db)
    queue_repo = QueueRepository(db)
    audit_repo = AuditRepository(db)

    user_service = UserManagementService(user_repo, conversation_repo, message_repo, handover_repo, audit_repo)
    project_service = ProjectManagementService(project_repo, project_faq_repo, marketing_plan_repo, project_prompt_repo, material_repo, user_repo)
    faq_service = ProjectFAQService(project_faq_repo)
    marketing_service = MarketingPlanService(marketing_plan_repo, marketing_plan_segment_repo)
    project_prompt_service = ProjectPromptService(project_prompt_repo)
    material_service = MaterialService(material_repo)
    persona_service = PersonaProfileService(persona_profile_repo, material_repo)
    daily_material_service = DailyMaterialService(material_repo)
    handover_service = HandoverService(handover_repo, user_repo)
    receipt_service = ReceiptService(receipt_repo)
    queue_service = QueueService(queue_repo)
    audit_service = AuditService(audit_repo)

    admin_api_service = AdminAPIService(
        user_service=user_service,
        project_service=project_service,
        faq_service=faq_service,
        marketing_service=marketing_service,
        project_prompt_service=project_prompt_service,
        material_service=material_service,
        persona_service=persona_service,
        daily_material_service=daily_material_service,
        handover_service=handover_service,
        receipt_service=receipt_service,
        queue_service=queue_service,
        audit_service=audit_service,
    )
    dashboard_service = DashboardService(user_repo, handover_repo, receipt_repo, queue_repo)
    admin_router = TGAdminCallbackRouter(admin_api_service, dashboard_service)

    understanding_engine = UserUnderstandingEngine()
    stage_engine = ConversationStageEngine()
    trajectory_engine = RelationshipTrajectoryEngine()
    rhythm_engine = RelationshipRhythmEngine()
    maturity_engine = RelationshipMaturityEngine()
    conflict_resolver = RhythmConflictResolver()
    memory_selector = MemorySelector(memory_repo)
    memory_priority_resolver = MemoryPriorityResolver()
    continuity_guard = ContinuityGuard()
    strategy_optimizer = AdaptiveStrategyOptimizer()
    turn_decision_engine = TurnDecisionEngine()
    project_window_evaluator = ProjectWindowEvaluator()
    project_nurture_planner = ProjectNurturePlanner()
    project_prompt_resolver = ProjectPromptResolver(project_prompt_repo)
    project_faq_resolver = ProjectFAQResolver(project_faq_repo)
    marketing_plan_resolver = MarketingPlanResolver(marketing_plan_repo, marketing_plan_segment_repo)
    persona_profile_resolver = PersonaProfileResolver(persona_profile_repo)
    material_retriever = MaterialRetriever(material_repo)
    material_ranker = MaterialRelevanceRanker()
    material_selection_planner = MaterialSelectionPlanner()
    humanization_controller = HumanizationController()
    reply_selfcheck_engine = ReplySelfCheckEngine()
    maturity_polish_engine = MaturityPolishEngine()
    memory_writeback_engine = MemoryWritebackEngine(memory_repo, conversation_repo)
    handover_learning_engine = HandoverLearningEngine(handover_repo, audit_repo)
    resume_bridge_planner = ResumeBridgePlanner()

    orchestrator = Orchestrator(
        user_repo=user_repo,
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        memory_repo=memory_repo,
        handover_repo=handover_repo,
        understanding_engine=understanding_engine,
        stage_engine=stage_engine,
        trajectory_engine=trajectory_engine,
        rhythm_engine=rhythm_engine,
        maturity_engine=maturity_engine,
        conflict_resolver=conflict_resolver,
        memory_selector=memory_selector,
        memory_priority_resolver=memory_priority_resolver,
        continuity_guard=continuity_guard,
        strategy_optimizer=strategy_optimizer,
        turn_decision_engine=turn_decision_engine,
        project_window_evaluator=project_window_evaluator,
        project_nurture_planner=project_nurture_planner,
        project_prompt_resolver=project_prompt_resolver,
        project_faq_resolver=project_faq_resolver,
        marketing_plan_resolver=marketing_plan_resolver,
        persona_profile_resolver=persona_profile_resolver,
        material_retriever=material_retriever,
        material_ranker=material_ranker,
        material_selection_planner=material_selection_planner,
        humanization_controller=humanization_controller,
        reply_selfcheck_engine=reply_selfcheck_engine,
        maturity_polish_engine=maturity_polish_engine,
        memory_writeback_engine=memory_writeback_engine,
        handover_learning_engine=handover_learning_engine,
        resume_bridge_planner=resume_bridge_planner,
    )

    return AppComponents(
        db=db,
        user_repo=user_repo,
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        project_repo=project_repo,
        project_prompt_repo=project_prompt_repo,
        project_faq_repo=project_faq_repo,
        marketing_plan_repo=marketing_plan_repo,
        marketing_plan_segment_repo=marketing_plan_segment_repo,
        material_repo=material_repo,
        persona_profile_repo=persona_profile_repo,
        memory_repo=memory_repo,
        handover_repo=handover_repo,
        receipt_repo=receipt_repo,
        queue_repo=queue_repo,
        audit_repo=audit_repo,
        admin_api_service=admin_api_service,
        dashboard_service=dashboard_service,
        admin_router=admin_router,
        orchestrator=orchestrator,
    )


async def _async_main() -> tuple[AppComponents, TelegramBotClient, OutboundSender, Flask]:
    logger.info("Starting stage 9 runtime application.")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is required")
    components = await build_app_components()
    await components.db.initialize_schema()
    bot_client = TelegramBotClient(BOT_TOKEN)
    if PUBLIC_BASE_URL and BOT_TOKEN:
        try:
            await bot_client.set_webhook(PUBLIC_BASE_URL, WEBHOOK_PATH, WEBHOOK_SECRET)
        except Exception:
            logger.exception("Failed to set webhook automatically.")
    outbound_sender = OutboundSender(bot_client)
    flask_app = create_flask_app(components, bot_client, outbound_sender)
    return components, bot_client, outbound_sender, flask_app


def main() -> None:
    components, _bot_client, _outbound_sender, flask_app = asyncio.run(_async_main())
    try:
        logger.info("Runtime app initialized successfully; serving Flask app.")
        flask_app.run(host=HOST, port=PORT, debug=False)
    finally:
        asyncio.run(components.db.close())


if __name__ == "__main__":
    main()
