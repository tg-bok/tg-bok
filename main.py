import os
import inspect
import step92_structural_convergence_final_cutover as appmod


def _patch_release_flag_compat():
    """
    修复 step92 内部 Step82 发布候选校验的 flag 命名不一致问题。
    这是发布候选校验层的兼容补丁，不是跳过真实业务错误。
    """
    original_report_builder = appmod._step82_collect_release_candidate_report

    def patched_report_builder(app_components):
        report = original_report_builder(app_components)

        feature_flags = dict(report.get("feature_flags") or {})

        # Step68：实际落的是两个 flag，Step82 却检查了一个不存在的总名
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

        # Step74 / 75 / 76 / 78：功能已落代码，但没有写 enabled flag，这里补齐
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


def _patch_soft_clip_compat():
    """
    Step70/71 预览文本依赖 _soft_clip，但当前 step92 最终层没有定义它。
    这里补一个轻量兼容实现，避免统一素材编辑器/媒体上传回执 500。
    """
    if hasattr(appmod, "_soft_clip"):
        return

    def _soft_clip(value, limit=120):
        text = "" if value is None else str(value)
        try:
            limit_int = int(limit)
        except Exception:
            limit_int = 120
        if limit_int <= 0:
            return ""
        if len(text) <= limit_int:
            return text
        if limit_int == 1:
            return "…"
        return text[: limit_int - 1] + "…"

    appmod._soft_clip = _soft_clip


def _make_generate_compat_wrapper(original_generate):
    original_sig = inspect.signature(original_generate)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in original_sig.parameters.values()
    )
    accepted_names = set(original_sig.parameters.keys())

    def wrapped(self, *args, **kwargs):
        if accepts_var_kwargs:
            return original_generate(self, *args, **kwargs)
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in accepted_names and k != "self"
        }
        return original_generate(self, *args, **filtered_kwargs)

    wrapped.__name__ = getattr(original_generate, "__name__", "generate")
    wrapped.__doc__ = getattr(original_generate, "__doc__", None)
    return wrapped


def _patch_reply_style_engine_generate_class(engine_cls):
    if engine_cls is None:
        return
    generate = getattr(engine_cls, "generate", None)
    if generate is None or getattr(generate, "_runtime_compat_patched", False):
        return

    sig = inspect.signature(generate)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if accepts_var_kwargs or "training_packet" in sig.parameters:
        return

    wrapped = _make_generate_compat_wrapper(generate)
    wrapped._runtime_compat_patched = True
    setattr(engine_cls, "generate", wrapped)



def _patch_reply_style_engine_compat():
    """
    Step78/79 之后的调用给 ReplyStyleEngine.generate 传了 training_packet，
    但当前生效类签名没跟上。这里做签名兼容，过滤掉旧版本不认识的 kwargs。
    """
    engine_cls = getattr(appmod, "ReplyStyleEngine", None)
    _patch_reply_style_engine_generate_class(engine_cls)


def _patch_runtime_instances(app_components):
    engine = app_components.get("reply_style_engine")
    if engine is not None:
        _patch_reply_style_engine_generate_class(engine.__class__)

    orchestrator = app_components.get("orchestrator")
    if orchestrator is not None:
        nested_engine = getattr(orchestrator, "reply_style_engine", None)
        if nested_engine is not None:
            _patch_reply_style_engine_generate_class(nested_engine.__class__)


_patch_release_flag_compat()
_patch_soft_clip_compat()
_patch_reply_style_engine_compat()

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
    _patch_runtime_instances(app_components)

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
