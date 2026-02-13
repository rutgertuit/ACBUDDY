import logging
import os

from flask import Flask

from app.config import Settings


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

    app.register_blueprint(health_bp)
    app.register_blueprint(webhook_bp)

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
