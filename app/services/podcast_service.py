"""ElevenLabs v3 TTS podcast generation.

Generates podcast audio by parsing a script into speaker turns and calling
the ElevenLabs TTS API with the eleven_v3 model per turn. Audio tags like
[laughs], [whispers], [excited] are natively supported by v3.

The resulting MP3 segments are concatenated and uploaded to GCS.
"""

import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elevenlabs.io/v1"

# Default voice IDs (ElevenLabs stock voices)
_DEFAULT_HOST_VOICE = "21m00Tcm4TlvDq8ikWAM"   # Rachel
_DEFAULT_GUEST_VOICE = "ErXwobaYiN019PkySvjV"  # Antoni

# Short silence MP3 frame (~0.3s) to insert between speaker turns.
# This is a valid MPEG audio frame of silence — avoids jarring cuts.
_SILENCE_MS = 300


def _headers(api_key: str) -> dict:
    return {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }


def _tts_v3(text: str, voice_id: str, api_key: str) -> bytes:
    """Generate speech for a single text segment using eleven_v3.

    Returns raw MP3 audio bytes.
    """
    url = f"{BASE_URL}/text-to-speech/{voice_id}"
    body = {
        "text": text,
        "model_id": "eleven_v3",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8,
            "style": 0.6,
            "use_speaker_boost": True,
        },
    }
    resp = requests.post(
        url,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


def _generate_silence(duration_ms: int, api_key: str) -> bytes:
    """Generate a short silence using the sound effects API.

    Falls back to an empty bytes object if it fails.
    """
    try:
        url = f"{BASE_URL}/sound-generation"
        body = {
            "text": "silence",
            "duration_seconds": duration_ms / 1000.0,
        }
        resp = requests.post(
            url,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content
    except Exception:
        return b""


def parse_script_turns(script: str) -> list[tuple[str, str]]:
    """Parse a podcast script into (speaker_name, text) turns.

    Expected format — each turn starts with 'SpeakerName:' at the start of a line.
    Handles multi-line turns (text continues until the next speaker label).
    Also handles LLM formatting quirks: **bold labels**, markdown fences, etc.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", script.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()

    # Strip all bold/italic markdown: **text** → text, *text* → text
    # This handles **Maya:**, **Professor Barnaby:**, etc.
    # Audio tags [excited] use brackets, not asterisks, so they're safe.
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", cleaned)

    # Also strip heading markers: ## Maya: → Maya:
    cleaned = re.sub(r"^#{1,4}\s*", "", cleaned, flags=re.MULTILINE)

    # Match lines starting with a speaker label like "Maya:" or "Professor Barnaby:"
    pattern = re.compile(r"^([A-Z][A-Za-z .0-9]+?):\s*", re.MULTILINE)
    turns: list[tuple[str, str]] = []

    splits = pattern.split(cleaned)
    # splits = [pre-text, speaker1, text1, speaker2, text2, ...]
    # Index 0 is any text before the first speaker label (usually empty)
    i = 1
    while i < len(splits) - 1:
        speaker = splits[i].strip()
        text = splits[i + 1].strip()
        if speaker and text:
            turns.append((speaker, text))
        i += 2

    return turns


def create_podcast(
    script: str,
    speaker_voices: dict[str, str],
    api_key: str,
    on_progress=None,
) -> bytes:
    """Generate podcast audio from a script using ElevenLabs v3 TTS.

    Args:
        script: Full podcast script with 'Speaker:' labels and v3 audio tags.
        speaker_voices: Dict mapping speaker name -> voice_id.
        api_key: ElevenLabs API key.
        on_progress: Optional callback(current_turn, total_turns) for status updates.

    Returns:
        Combined MP3 audio bytes.
    """
    turns = parse_script_turns(script)
    if not turns:
        # Log first 500 chars to help debug parsing failures
        logger.error("Script turn parsing failed. First 500 chars:\n%s", script[:500])
        raise ValueError("Could not parse any speaker turns from the script")

    logger.info("Podcast: %d turns, speakers: %s", len(turns), list(speaker_voices.keys()))

    # Determine default voices for unknown speakers
    known_voices = list(speaker_voices.values())
    default_voices = [_DEFAULT_HOST_VOICE, _DEFAULT_GUEST_VOICE]

    audio_segments: list[bytes] = []
    total = len(turns)

    for idx, (speaker, text) in enumerate(turns):
        if on_progress:
            on_progress(idx + 1, total)

        # Resolve voice for this speaker
        voice_id = speaker_voices.get(speaker, "")
        if not voice_id:
            # Assign alternating defaults for unknown speakers
            voice_id = default_voices[idx % 2] if not known_voices else known_voices[idx % len(known_voices)]

        logger.info("Podcast turn %d/%d: %s (%d chars)", idx + 1, total, speaker, len(text))

        # Generate audio for this turn
        retries = 2
        for attempt in range(retries + 1):
            try:
                audio = _tts_v3(text, voice_id, api_key)
                audio_segments.append(audio)
                break
            except requests.exceptions.HTTPError as e:
                if attempt < retries and e.response is not None and e.response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning("Rate limited on turn %d, waiting %ds", idx + 1, wait)
                    time.sleep(wait)
                else:
                    raise

    # Concatenate all MP3 segments
    combined = b"".join(audio_segments)
    logger.info("Podcast audio generated: %d turns, %.1f MB", len(turns), len(combined) / 1024 / 1024)
    return combined


def upload_podcast_audio(audio_bytes: bytes, job_id: str, bucket_name: str) -> str:
    """Upload podcast MP3 to GCS and return its public URL."""
    if not bucket_name:
        return ""

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_name = f"results/{job_id}_podcast.mp3"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(audio_bytes, content_type="audio/mpeg")

        return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    except Exception:
        logger.exception("Failed to upload podcast audio to GCS")
        return ""


def list_voices(api_key: str) -> list[dict]:
    """List available ElevenLabs voices."""
    url = f"{BASE_URL}/voices"
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    voices = data.get("voices", data if isinstance(data, list) else [])
    return [{"voice_id": v.get("voice_id", ""), "name": v.get("name", "")} for v in voices]
