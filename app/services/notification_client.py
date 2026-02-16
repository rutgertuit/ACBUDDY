"""Notification client — sends alerts when watch changes are detected.

Supports webhook (Slack/Discord/custom) and email (SendGrid) notifications.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)


async def send_watch_notification(watch, update) -> None:
    """Send notification when a watch detects changes."""
    if watch.notification_email:
        try:
            await _send_email(watch.notification_email, watch.query, update.summary)
        except Exception:
            logger.exception("Email notification failed for watch %s", watch.id)

    if watch.notification_webhook:
        try:
            await _send_webhook(watch.notification_webhook, watch, update)
        except Exception:
            logger.exception("Webhook notification failed for watch %s", watch.id)


async def _send_email(to: str, subject_topic: str, body: str) -> None:
    """Send notification email via SendGrid."""
    api_key = os.getenv("SENDGRID_API_KEY", "")
    if not api_key:
        logger.warning("SendGrid API key not configured, skipping email notification")
        return

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "personalizations": [{"to": [{"email": to}]}],
                "from": {"email": os.getenv("SENDGRID_FROM_EMAIL", "noreply@luminary.app")},
                "subject": f"Luminary Watch Alert: {subject_topic[:80]}",
                "content": [
                    {
                        "type": "text/plain",
                        "value": (
                            f"Changes detected for your research watch:\n\n"
                            f"Topic: {subject_topic}\n\n"
                            f"Summary of changes:\n{body}\n\n"
                            f"— Luminary Research Intelligence"
                        ),
                    }
                ],
            },
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("Email notification sent to %s for topic: %s", to, subject_topic[:60])


async def _send_webhook(url: str, watch, update) -> None:
    """POST JSON to a webhook URL (Slack/Discord/custom)."""
    payload = {
        "text": f"Luminary Watch Alert: Changes detected for *{watch.query}*",
        "watch_id": watch.id,
        "query": watch.query,
        "changed": update.changed,
        "summary": update.summary,
        "checked_at": update.checked_at,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        logger.info("Webhook notification sent to %s for watch %s", url[:60], watch.id)
