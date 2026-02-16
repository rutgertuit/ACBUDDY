"""Static metadata for the ElevenLabs conversation agents."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    slug: str
    name: str
    subtitle: str
    personality: str
    icon: str
    color: str  # Tailwind color keyword used in the UI
    voice_id: str = ""  # ElevenLabs voice ID for podcast generation


AGENTS: dict[str, AgentProfile] = {
    "maya": AgentProfile(
        slug="maya",
        name="Maya",
        subtitle="The Zero-Filter Lead Analyst",
        personality="Sharp, caffeinated, dry humor, no fluff",
        icon="bolt",
        color="cyan",
        voice_id="xBPWZQ2gzFKIn63cFXfT",
    ),
    "barnaby": AgentProfile(
        slug="barnaby",
        name="Professor Barnaby",
        subtitle="The Chaos Academic",
        personality="Jack Black energy, explosive enthusiasm, sound effects",
        icon="science",
        color="amber",
        voice_id="xL9fhtOiTXXCASQKlBiH",
    ),
    "consultant": AgentProfile(
        slug="consultant",
        name="Consultant 4.0",
        subtitle="Senior Partner (Beta)",
        personality="McKinsey polish + malfunctioning Humanity Patch",
        icon="business_center",
        color="violet",
        voice_id="DdjCTAxRdHgBQaM7jniZ",
    ),
    "rutger": AgentProfile(
        slug="rutger",
        name="Rutger",
        subtitle="The Creative Technologist",
        personality="Jazz-swing duality, storytelling visionary, Dutch pragmatism meets Google-scale vision",
        icon="equalizer",
        color="emerald",
        voice_id="3UDL95XpjDt8BT7uFojh",
    ),
}


def get_agent_id(slug: str, settings) -> str:
    """Return the ElevenLabs agent ID for a given slug, or empty string."""
    mapping = {
        "maya": settings.elevenlabs_agent_id_maya,
        "barnaby": settings.elevenlabs_agent_id_barnaby,
        "consultant": settings.elevenlabs_agent_id_consultant,
        "rutger": settings.elevenlabs_agent_id_rutger,
    }
    return mapping.get(slug, "")


def get_voice_id(slug: str, settings) -> str:
    """Return the podcast voice ID for a given agent slug.

    Checks env var overrides first, then falls back to the voice_id
    baked into the AgentProfile.
    """
    override = {
        "maya": settings.podcast_voice_id_maya,
        "barnaby": settings.podcast_voice_id_barnaby,
        "consultant": settings.podcast_voice_id_consultant,
        "rutger": settings.podcast_voice_id_rutger,
    }.get(slug, "")
    if override:
        return override
    profile = AGENTS.get(slug)
    return profile.voice_id if profile else ""
