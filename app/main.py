import logging
import os
import signal
import sys

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

    # Register SIGTERM handler for graceful shutdown
    _setup_sigterm_handler(app)

    app.logger.info("Luminary started (environment=%s)", settings.environment)
    return app


def _setup_sigterm_handler(app: Flask) -> None:
    """Register SIGTERM handler to checkpoint running DEEP jobs before shutdown.

    Cloud Run sends SIGTERM when recycling instances. We get ~10s to save state.
    This marks running DEEP jobs as interrupted in GCS metadata so the UI can
    offer a resume button instead of silently losing the job.
    """
    def _on_sigterm(signum, frame):
        logger = logging.getLogger(__name__)
        logger.info("SIGTERM received — checkpointing running DEEP jobs")

        try:
            from app.services.job_tracker import get_running_deep_jobs, JobStatus, update_job
            from app.services import gcs_client

            settings = app.config.get("SETTINGS")
            bucket = settings.gcs_results_bucket if settings else ""
            deep_jobs = get_running_deep_jobs()

            for job in deep_jobs:
                try:
                    # Mark in-memory so any final poll returns "failed" instead of "running"
                    update_job(job.job_id,
                               status=JobStatus.FAILED,
                               error="Server shutdown during research. Resume available.")
                    # Update GCS metadata so archive shows it as interrupted
                    if bucket:
                        gcs_client.update_metadata(job.job_id, bucket, {
                            "status": "interrupted",
                            "error": "Server shutdown during research (SIGTERM). Resume available.",
                        })
                    logger.info("Marked job %s as interrupted", job.job_id)
                except Exception:
                    logger.exception("Failed to mark job %s as interrupted", job.job_id)
        except Exception:
            logger.exception("SIGTERM handler error")

        # Re-raise to let gunicorn handle normal shutdown
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)


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
