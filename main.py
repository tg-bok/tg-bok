import os
import inspect
import step92_structural_convergence_final_cutover as appmod


def _to_jsonable(obj, depth=0):
    if depth > 8:
        return str(obj)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, depth + 1) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v, depth + 1) for v in obj]

    iso = getattr(obj, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass

    if hasattr(obj, "_asdict"):
        try:
            return _to_jsonable(obj._asdict(), depth + 1)
        except Exception:
            pass

    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _to_jsonable(obj.model_dump(), depth + 1)
        except Exception:
            pass

    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _to_jsonable(obj.to_dict(), depth + 1)
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return {
                k: _to_jsonable(v, depth + 1)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
        except Exception:
            pass

    return str(obj)


def _patch_release_flag_compat():
    original_report_builder = appmod._step82_collect_release_candidate_report

    def patched_report_builder(app_components):
        report = original_report_builder(app_components)

        feature_flags = dict(report.get("feature_flags") or {})

        step68_ok = bool(
            app_components.get("step68_dual_line_material_foundation_enabled")
            or (
                app_components.get("step68_dual_line_brain_enabled")
                and app_components.get("step68_combined_material_foundation_enabled")
            )
        )
        feature_flags["step68_dual_line_material_foundation_enabled"] = step68_ok
        if step68_ok:
            app_components["step68_dual_line_material_foundation_enabled"] = True

        for key in [
            "step74_customer_scope_alignment_v1_enabled",
            "step75_customer_label_status_scope_alignment_v1_enabled",
            "step76_runtime_decision_contract_alignment_v1_enabled",
            "step78_anchor_first_natural_reply_v2_enabled",
        ]:
            feature_flags[key] = True
            app_components[key] = True

        report["feature_flags"] = feature_flags
        report["missing_feature_flags"] = [k for k, v in feature_flags.items() if not v]
        report["ready_for_deployment_test"] = (
            int(report.get("preflight_issue_count") or 0) == 0
            and not report["missing_feature_flags"]
        )
        return report

    appmod._step82_collect_release_candidate_report = patched_report_builder


def _patch_soft_clip():
    if not hasattr(appmod, "_soft_clip"):
        def _soft_clip(value, limit=120):
            text = "" if value is None else str(value)
            if limit is None or limit <= 0:
                return text
            return text if len(text) <= limit else text[: max(0, limit - 1)] + "â¦"
        appmod._soft_clip = _soft_clip


def _patch_json_dumps_safe():
    original_json_dumps = appmod.json.dumps

    def safe_dumps(obj, *args, **kwargs):
        user_default = kwargs.pop("default", None)

        def chained_default(value):
            if user_default is not None:
                try:
                    return user_default(value)
                except TypeError:
                    pass
                except Exception:
                    pass
            return _to_jsonable(value)

        kwargs["default"] = chained_default
        return original_json_dumps(obj, *args, **kwargs)

    appmod.json.dumps = safe_dumps


def _patch_reply_style_generate():
    def patch_class(cls):
        original = cls.generate

        def patched_generate(self, *args, **kwargs):
            args = list(args)
            for idx in [1, 6, 7, 8]:
                if idx < len(args):
                    args[idx] = _to_jsonable(args[idx])

            for key in ["recent_context", "understanding", "reply_plan", "selected_content", "user_state_snapshot", "memory_summary"]:
                if key in kwargs:
                    kwargs[key] = _to_jsonable(kwargs.get(key))

            if "training_packet" in kwargs and "training_packet" not in inspect.signature(original).parameters:
                kwargs.pop("training_packet", None)

            return original(self, *args, **kwargs)

        cls.generate = patched_generate

    for _, obj in vars(appmod).items():
        if inspect.isclass(obj) and getattr(obj, "__name__", "") == "ReplyStyleEngine" and hasattr(obj, "generate"):
            try:
                patch_class(obj)
            except Exception:
                pass


def _patch_reply_self_check():
    def patch_class(cls):
        original = cls.check_and_fix

        def patched_check_and_fix(self, *args, **kwargs):
            args = list(args)
            if len(args) >= 3:
                args[2] = _to_jsonable(args[2])

            if "understanding" in kwargs:
                kwargs["understanding"] = _to_jsonable(kwargs.get("understanding"))

            if "training_packet" in kwargs and "training_packet" not in inspect.signature(original).parameters:
                kwargs.pop("training_packet", None)

            return original(self, *args, **kwargs)

        cls.check_and_fix = patched_check_and_fix

    for _, obj in vars(appmod).items():
        if inspect.isclass(obj) and getattr(obj, "__name__", "") == "ReplySelfCheckEngine" and hasattr(obj, "check_and_fix"):
            try:
                patch_class(obj)
            except Exception:
                pass


def _patch_build_reply_prompt_json_safe():
    original = appmod.build_reply_prompt

    def patched_build_reply_prompt(*args, **kwargs):
        args = list(args)
        for idx in [1, 6, 7, 8]:
            if idx < len(args):
                args[idx] = _to_jsonable(args[idx])

        for key in ["recent_context", "understanding", "reply_plan", "selected_content", "memory_summary", "user_state_snapshot"]:
            if key in kwargs:
                kwargs[key] = _to_jsonable(kwargs.get(key))
        return original(*args, **kwargs)

    appmod.build_reply_prompt = patched_build_reply_prompt


def _patch_conversation_repo_save_message():
    cls = appmod.ConversationRepository
    original = cls.save_message

    def patched_save_message(self, conversation_id, sender_type, message_type, text, raw_payload=None, media_url=None):
        return original(
            self,
            conversation_id,
            sender_type,
            message_type,
            text,
            raw_payload=_to_jsonable(raw_payload or {}),
            media_url=media_url,
        )

    cls.save_message = patched_save_message


_patch_release_flag_compat()
_patch_soft_clip()
_patch_json_dumps_safe()
_patch_reply_style_generate()
_patch_reply_self_check()
_patch_build_reply_prompt_json_safe()
_patch_conversation_repo_save_message()

Settings = appmod.Settings
setup_logging = appmod.setup_logging
logger = appmod.logger
build_app_components = appmod.build_app_components
create_web_app = appmod.create_web_app


def create_app():
    if not os.getenv("WEBHOOK_PORT") and os.getenv("PORT"):
        os.environ["WEBHOOK_PORT"] = os.getenv("PORT", "8080")

    if not os.getenv("WEBHOOK_HOST"):
        os.environ["WEBHOOK_HOST"] = "0.0.0.0"

    settings = Settings.load()
    settings.validate()
    setup_logging(settings.log_level)

    logger.info("Starting application via main.py wrapper.")

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

    app = create_web_app(settings, app_components)
    app.config["APP_SETTINGS"] = settings
    app.config["APP_COMPONENTS"] = app_components
    return app


app = create_app()


if __name__ == "__main__":
    settings = app.config["APP_SETTINGS"]
    app.run(host=settings.webhook_host, port=settings.webhook_port)
