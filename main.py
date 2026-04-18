import os

from step92_structural_convergence_final_cutover import (
    Settings,
    setup_logging,
    logger,
    build_app_components,
    create_web_app,
)


def create_app():
    # Railway 会自动给 PORT，这里兼容一下
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
