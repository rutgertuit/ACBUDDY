"""ElevenLabs Studio Podcasts API integration for generating audio from research."""

import logging
import time

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elevenlabs.io/v1"


def _headers(api_key: str) -> dict:
    return {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }


# Style-specific instructions for the ElevenLabs GenFM engine
_STYLE_INSTRUCTIONS = {
    "executive": (
        "Two senior analysts discuss strategic insights from research findings. "
        "Professional, data-driven, concise. Reference specific data points and "
        "strategic implications. Keep the tone authoritative but accessible."
    ),
    "curious": (
        "An enthusiastic host interviews a knowledgeable expert about the research. "
        "Educational, accessible, use analogies and real-world examples. "
        "The host asks clarifying questions that the audience would want answered."
    ),
    "debate": (
        "Two experts present different angles and challenge each other's assumptions. "
        "Balanced, thought-provoking, intellectually rigorous. Each speaker brings "
        "unique perspectives while remaining respectful and evidence-based."
    ),
}


# Default voice IDs (ElevenLabs stock voices)
_DEFAULT_HOST_VOICE = "21m00Tcm4TlvDq8ikWAM"   # Rachel
_DEFAULT_GUEST_VOICE = "ErXwobaYiN019PkySvjV"  # Antoni


def create_podcast(
    script: str,
    style: str,
    api_key: str,
    host_voice_id: str = "",
    guest_voice_id: str = "",
) -> str:
    """Submit a podcast script to ElevenLabs Studio Podcasts API.

    Returns the project_id for polling status.
    """
    url = f"{BASE_URL}/studio/podcasts"
    instructions = _STYLE_INSTRUCTIONS.get(style, _STYLE_INSTRUCTIONS["curious"])

    host_vid = host_voice_id or _DEFAULT_HOST_VOICE
    guest_vid = guest_voice_id or _DEFAULT_GUEST_VOICE

    body = {
        "model_id": "eleven_multilingual_v2",
        "mode": {
            "type": "conversation",
            "conversation": {
                "host_voice_id": host_vid,
                "guest_voice_id": guest_vid,
            },
        },
        "source": {
            "type": "text",
            "text": script,
        },
        "quality_preset": "standard",
        "duration_scale": "default",
        "instructions_prompt": instructions,
    }

    logger.info("Creating podcast (style=%s, script_len=%d, host=%s, guest=%s)", style, len(script), host_vid, guest_vid)
    resp = requests.post(url, headers=_headers(api_key), json=body, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    project_id = result.get("project_id", result.get("id", ""))
    logger.info("Podcast created: project_id=%s", project_id)
    return project_id


def get_podcast_status(project_id: str, api_key: str) -> dict:
    """Get the status of a podcast project.

    Returns dict with at least {status, ...}.
    Status values: 'creating', 'processing', 'completed', 'failed'.
    """
    url = f"{BASE_URL}/studio/podcasts/{project_id}"
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_podcast_audio_url(project_id: str, api_key: str) -> str:
    """Get the audio stream URL for a completed podcast.

    Chains: project -> chapters -> snapshots -> audio stream.
    Returns the download/stream URL.
    """
    headers = _headers(api_key)

    # Get project to find chapters
    project = get_podcast_status(project_id, api_key)

    # Try direct audio URL from project
    if project.get("audio_url"):
        return project["audio_url"]

    # Get chapters
    chapters_url = f"{BASE_URL}/studio/podcasts/{project_id}/chapters"
    resp = requests.get(chapters_url, headers=headers, timeout=30)
    resp.raise_for_status()
    chapters = resp.json()

    # Handle response format — could be list or dict with items
    chapter_list = chapters if isinstance(chapters, list) else chapters.get("chapters", chapters.get("items", []))
    if not chapter_list:
        logger.warning("No chapters found for podcast %s", project_id)
        return ""

    chapter_id = chapter_list[0].get("chapter_id", chapter_list[0].get("id", ""))
    if not chapter_id:
        return ""

    # Get snapshots for the chapter
    snapshots_url = f"{BASE_URL}/studio/podcasts/{project_id}/chapters/{chapter_id}/snapshots"
    resp = requests.get(snapshots_url, headers=headers, timeout=30)
    resp.raise_for_status()
    snapshots = resp.json()

    snapshot_list = snapshots if isinstance(snapshots, list) else snapshots.get("snapshots", snapshots.get("items", []))
    if not snapshot_list:
        logger.warning("No snapshots found for podcast %s chapter %s", project_id, chapter_id)
        return ""

    snapshot_id = snapshot_list[0].get("snapshot_id", snapshot_list[0].get("id", ""))
    if not snapshot_id:
        return ""

    # The stream URL
    stream_url = (
        f"{BASE_URL}/studio/podcasts/{project_id}/chapters/{chapter_id}"
        f"/snapshots/{snapshot_id}/stream"
    )
    return stream_url


def poll_until_complete(
    project_id: str,
    api_key: str,
    poll_interval: int = 10,
    max_wait: int = 600,
) -> dict:
    """Poll podcast status until completed or failed.

    Returns the final status dict.
    """
    elapsed = 0
    while elapsed < max_wait:
        status = get_podcast_status(project_id, api_key)
        state = status.get("status", "unknown")
        logger.info("Podcast %s status: %s (elapsed=%ds)", project_id, state, elapsed)

        if state in ("completed", "done"):
            return status
        if state in ("failed", "error"):
            raise RuntimeError(f"Podcast generation failed: {status.get('error', 'unknown error')}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Podcast {project_id} did not complete within {max_wait}s")


def list_voices(api_key: str) -> list[dict]:
    """List available ElevenLabs voices."""
    url = f"{BASE_URL}/voices"
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    voices = data.get("voices", data if isinstance(data, list) else [])
    return [{"voice_id": v.get("voice_id", ""), "name": v.get("name", "")} for v in voices]
