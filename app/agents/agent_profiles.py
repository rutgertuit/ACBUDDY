"""Static metadata for the 3 ElevenLabs conversation agents."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    slug: str
    name: str
    subtitle: str
    personality: str
    icon: str
    color: str  # Tailwind color keyword used in the UI


AGENTS: dict[str, AgentProfile] = {
    "maya": AgentProfile(
        slug="maya",
        name="Maya",
        subtitle="The Zero-Filter Lead Analyst",
        personality="Sharp, caffeinated, dry humor, no fluff",
        icon="bolt",
        color="cyan",
    ),
    "barnaby": AgentProfile(
        slug="barnaby",
        name="Professor Barnaby",
        subtitle="The Chaos Academic",
        personality="Jack Black energy, explosive enthusiasm, sound effects",
        icon="science",
        color="amber",
    ),
    "consultant": AgentProfile(
        slug="consultant",
        name="Consultant 4.0",
        subtitle="Senior Partner (Beta)",
        personality="McKinsey polish + malfunctioning Humanity Patch",
        icon="business_center",
        color="violet",
    ),
}


def get_agent_id(slug: str, settings) -> str:
    """Return the ElevenLabs agent ID for a given slug, or empty string."""
    mapping = {
        "maya": settings.elevenlabs_agent_id_maya,
        "barnaby": settings.elevenlabs_agent_id_barnaby,
        "consultant": settings.elevenlabs_agent_id_consultant,
    }
    return mapping.get(slug, "")
