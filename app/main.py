import logging
import os

from flask import Flask

from app.config import Settings


def _patch_adk_telemetry():
    """Monkey-patch ADK telemetry to prevent bytes serialization crashes.

    ADK's trace_call_llm calls json.dumps on the LLM request, which crashes
    when tool results contain bytes (e.g., from fetched binary URLs). This
    wraps the function in a try/except so telemetry failures are silenced.
    Patches both the module attribute and the local binding in base_llm_flow.
    """
    try:
        from google.adk import telemetry

        _original = telemetry.trace_call_llm

        def _safe_trace_call_llm(*args, **kwargs):
            try:
                return _original(*args, **kwargs)
            except (TypeError, ValueError):
                pass  # silently skip trace on serialization errors

        telemetry.trace_call_llm = _safe_trace_call_llm

        # Also patch the local binding in base_llm_flow (uses from-import)
        try:
            from google.adk.flows.llm_flows import base_llm_flow
            base_llm_flow.trace_call_llm = _safe_trace_call_llm
        except Exception:
            pass
    except Exception:
        pass  # ADK not installed or different version


_patch_adk_telemetry()


def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(__name__)

    # Configure logging
    _setup_logging(app)

    # Load settings
    settings = Settings()
    app.config["SETTINGS"] = settings

    # Register blueprints
    from app.routes.health import health_bp
    from app.routes.webhook import webhook_bp
    from app.routes.ui_api import ui_api_bp
    from app.routes.explore import explore_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(webhook_bp)
    app.register_blueprint(ui_api_bp)
    app.register_blueprint(explore_bp)

    app.logger.info("ACBUDDY started (environment=%s)", settings.environment)
    return app


def _setup_logging(app: Flask) -> None:
    """Configure structured logging for Cloud Run or standard logging locally."""
    environment = os.getenv("ENVIRONMENT", "local")

    if environment != "local":
        try:
            import google.cloud.logging

            client = google.cloud.logging.Client()
            client.setup_logging()
            app.logger.info("Cloud Logging configured")
        except Exception:
            logging.basicConfig(level=logging.INFO)
            app.logger.warning("Failed to set up Cloud Logging, using basic logging")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
