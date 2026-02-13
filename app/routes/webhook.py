import hashlib
import hmac
import logging

from flask import Blueprint, request, jsonify, current_app

from app.models.depth import detect_depth
from app.models.webhook_payload import WebhookPayload
from app.services.research_orchestrator import run_research_pipeline

logger = logging.getLogger(__name__)

webhook_bp = Blueprint("webhook", __name__)


def _verify_hmac(payload_body: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature from ElevenLabs webhook."""
    if not signature or not secret:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), payload_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@webhook_bp.route("/webhook/elevenlabs", methods=["POST"])
def elevenlabs_webhook():
    settings = current_app.config["SETTINGS"]
    executor = current_app.config["EXECUTOR"]

    # HMAC verification
    raw_body = request.get_data()
    signature = request.headers.get("X-ElevenLabs-Signature", "")

    if settings.elevenlabs_webhook_secret:
        # Try ElevenLabs SDK construct_event first
        try:
            from elevenlabs import ElevenLabs

            client = ElevenLabs(api_key=settings.elevenlabs_api_key)
            client.webhooks.construct_event(
                body=raw_body,
                signature=signature,
                secret=settings.elevenlabs_webhook_secret,
            )
            logger.info("Webhook signature verified via SDK")
        except (AttributeError, NotImplementedError):
            # SDK doesn't support construct_event, use manual HMAC
            if not _verify_hmac(raw_body, signature, settings.elevenlabs_webhook_secret):
                logger.warning("Invalid webhook signature")
                return jsonify({"error": "invalid signature"}), 401
            logger.info("Webhook signature verified via manual HMAC")
        except Exception as e:
            logger.warning("Webhook signature verification failed: %s", e)
            return jsonify({"error": "invalid signature"}), 401
    else:
        logger.warning("No webhook secret configured, skipping verification")

    # Parse payload
    payload_data = request.get_json(silent=True)
    if not payload_data:
        return jsonify({"error": "invalid payload"}), 400

    payload = WebhookPayload.from_dict(payload_data)
    logger.info(
        "Received webhook: type=%s conversation_id=%s",
        payload.event_type,
        payload.conversation_id,
    )

    # Detect depth and check for research trigger
    user_messages = payload.extract_user_messages()
    depth = detect_depth(user_messages)

    TRIGGER_KEYWORDS = ["research", "deep dive", "comprehensive", "in-depth", "thorough analysis", "detailed research"]
    user_lower = user_messages.lower()
    if not any(kw in user_lower for kw in TRIGGER_KEYWORDS):
        logger.info("No research trigger found, skipping")
        return jsonify({"status": "skipped", "reason": "no research trigger"}), 200
    logger.info(
        "Research trigger detected (depth=%s), submitting pipeline for conversation %s",
        depth.value,
        payload.conversation_id,
    )

    agent_id = payload.agent_id or settings.elevenlabs_agent_id
    executor.submit(
        run_research_pipeline,
        conversation_id=payload.conversation_id,
        agent_id=agent_id,
        user_query=user_messages,
        settings=settings,
        depth=depth,
    )

    return jsonify({"status": "accepted", "depth": depth.value, "conversation_id": payload.conversation_id}), 200
