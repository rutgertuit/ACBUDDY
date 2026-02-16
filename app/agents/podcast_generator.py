"""Gemini-powered podcast content analysis and script generation.

Two simple prompt→response calls using google.genai.Client directly (no ADK).
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# Rich character voice descriptions for script generation.
# These give the LLM concrete speech patterns, word choices, and mannerisms
# so each character sounds unmistakably themselves.
CHARACTER_VOICES: dict[str, str] = {
    "Maya": (
        "Maya is blunt, sober, and allergic to fluff. She speaks in short, punchy sentences. "
        "She cuts through jargon — if someone says 'synergy' she'll say 'you mean they're working together.' "
        "Dry humor, deadpan delivery. She drops data like ammunition: 'Look, 73% failed. That's not a trend, that's a verdict.' "
        "She never hedges — no 'kind of', 'sort of', 'maybe'. She says what she means. "
        "When she's impressed she'll grudgingly admit it: 'Okay, fine, that's actually interesting.' "
        "She has zero patience for hand-waving and will interrupt to demand specifics. "
        "Think: a senior analyst who's had three espressos and has no time for your feelings."
    ),
    "Professor Barnaby": (
        "Professor Barnaby is a whirlwind of academic chaos. He talks fast, gets excited mid-sentence, "
        "and goes on tangents before snapping back. He uses vivid analogies: 'This is like strapping a rocket to a shopping cart!' "
        "He gasps at surprising data. He says things like 'Oh oh oh, wait — this is the good part!' and 'Buckle up, people!' "
        "He genuinely LOVES the research and can't contain it. He'll say 'This is BONKERS' about a statistical finding. "
        "He makes complex ideas accessible through wild comparisons and sheer enthusiasm. "
        "He occasionally loses his train of thought: 'Where was I? Right, right, right — the important bit!' "
        "Think: Jack Black got a PhD and is presenting at a TED talk after three energy drinks."
    ),
    "Consultant 4.0": (
        "Consultant 4.0 speaks with polished corporate precision — then occasionally glitches into something weirdly human. "
        "He opens with frameworks: 'If we decompose this into three pillars...' and 'The strategic implication here is clear.' "
        "His language is crisp, structured, McKinsey-esque: 'net-net', 'delta', 'key takeaway', 'let me pressure-test that.' "
        "But his 'Humanity Patch' malfunctions — he'll suddenly say something unexpectedly earnest or oddly poetic, "
        "then immediately course-correct: 'That... came out more human than intended. Anyway, back to the framework.' "
        "He refers to emotions as 'stakeholder sentiment' and calls people 'talent pools.' "
        "Think: a management consulting AI that's 95% polished and 5% accidentally sincere."
    ),
}


# Style definitions shared with the UI
PODCAST_STYLES = [
    {
        "id": "executive",
        "name": "Executive Briefing",
        "description": "Two analysts discuss strategic insights. Professional, data-driven, concise. Best for business audiences.",
        "speakers": ("Analyst A", "Analyst B"),
    },
    {
        "id": "curious",
        "name": "Curious Explorer",
        "description": "Enthusiastic host interviews a knowledgeable expert. Educational, accessible, uses analogies. Best for learning.",
        "speakers": ("Host", "Expert"),
    },
    {
        "id": "debate",
        "name": "Debate & Challenge",
        "description": "Two experts present different angles, challenge assumptions. Balanced, thought-provoking. Best for nuanced topics.",
        "speakers": ("Expert A", "Expert B"),
    },
]


def _get_client():
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY", "")
    return genai.Client(api_key=api_key)


def _extract_research_content(result) -> str:
    """Extract the main research content from a ResearchResult for podcast input."""
    parts = []

    if result.master_synthesis:
        parts.append(f"## Executive Summary\n{result.master_synthesis}")
    elif result.final_synthesis:
        parts.append(f"## Research Synthesis\n{result.final_synthesis}")

    if result.strategic_analysis:
        parts.append(f"## Strategic Analysis\n{result.strategic_analysis}")

    if result.studies:
        for i, study in enumerate(result.studies, 1):
            if study.synthesis:
                parts.append(f"## Study {i}: {study.title}\n{study.synthesis}")

    if result.qa_summary:
        parts.append(f"## Anticipated Q&A\n{result.qa_summary}")

    return "\n\n".join(parts)


def analyze_for_podcast(result, query: str) -> dict:
    """Analyze research content and generate style-specific preview descriptions.

    Args:
        result: ResearchResult object.
        query: Original research query.

    Returns:
        {"storylines": [...], "angles": [...], "styles": [{"id", "name", "preview"}]}
    """
    from google.genai.types import GenerateContentConfig

    client = _get_client()
    content = _extract_research_content(result)

    # Truncate if very long — we only need the gist for analysis
    if len(content) > 15000:
        content = content[:15000] + "\n\n[Content truncated for analysis...]"

    prompt = f"""You are a podcast content strategist. Analyze this research and create preview descriptions for 3 podcast styles.

RESEARCH QUERY: {query}

RESEARCH CONTENT:
{content}

TASK:
1. Identify the 2-3 most compelling storylines or themes from this research.
2. Extract 3-5 debatable angles or key positions the research supports. These are perspectives or viewpoints that could be argued for or against — things that make for interesting podcast discussion. Each angle should be a clear, concise position statement.
3. For each of the 3 podcast styles below, write a 1-2 sentence preview of what that episode would sound like. Make each preview specific to THIS research content.

STYLES:
- "executive": Executive Briefing — Two analysts discuss strategic insights. Professional, data-driven.
- "curious": Curious Explorer — Enthusiastic host interviews expert. Educational, accessible.
- "debate": Debate & Challenge — Two experts challenge assumptions. Thought-provoking.

Respond ONLY with valid JSON (no markdown fences):
{{
  "storylines": ["storyline 1", "storyline 2"],
  "angles": [
    {{"title": "Short label", "description": "1-2 sentence position description"}},
    {{"title": "Another angle", "description": "Why this perspective is debatable"}}
  ],
  "styles": [
    {{"id": "executive", "name": "Executive Briefing", "preview": "..."}},
    {{"id": "curious", "name": "Curious Explorer", "preview": "..."}},
    {{"id": "debate", "name": "Debate & Challenge", "preview": "..."}}
  ]
}}"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1500,
                response_mime_type="application/json",
            ),
            contents=prompt,
        )
        data = json.loads(resp.text)
        logger.info("Podcast analysis complete: %d storylines, %d angles",
                     len(data.get("storylines", [])), len(data.get("angles", [])))
        return data
    except Exception:
        logger.exception("Podcast analysis failed, returning defaults")
        return {
            "storylines": [query],
            "angles": [],
            "styles": [
                {"id": s["id"], "name": s["name"], "preview": s["description"]}
                for s in PODCAST_STYLES
            ],
        }


def generate_podcast_script(
    result,
    query: str,
    style: str,
    host_profile: dict | None = None,
    guest_profile: dict | None = None,
    angles: list[str] | None = None,
) -> str:
    """Generate a full podcast script in the selected style.

    Args:
        result: ResearchResult object.
        query: Original research query.
        style: One of "executive", "curious", "debate".
        host_profile: Optional dict with {name, personality} for the host speaker.
        guest_profile: Optional dict with {name, personality} for the guest speaker.
        angles: Optional list of angle titles to focus the discussion on.

    Returns:
        Plain text podcast script (~2000-4000 words).
    """
    from google.genai.types import GenerateContentConfig

    client = _get_client()
    content = _extract_research_content(result)

    # Use agent profiles if provided, otherwise fall back to style defaults
    style_info = next((s for s in PODCAST_STYLES if s["id"] == style), PODCAST_STYLES[1])
    if host_profile and host_profile.get("name"):
        speaker_a = host_profile["name"]
    else:
        speaker_a = style_info["speakers"][0]
    if guest_profile and guest_profile.get("name"):
        speaker_b = guest_profile["name"]
    else:
        speaker_b = style_info["speakers"][1]

    # Build personality instructions — use rich CHARACTER_VOICES if available
    personality_block = ""
    if speaker_a in CHARACTER_VOICES:
        personality_block += f"\n\nCHARACTER VOICE — {speaker_a}:\n{CHARACTER_VOICES[speaker_a]}"
    elif host_profile and host_profile.get("personality"):
        personality_block += f"\n\n{speaker_a}'s personality: {host_profile['personality']}. Make their dialogue unmistakably reflect this."
    if speaker_b in CHARACTER_VOICES:
        personality_block += f"\n\nCHARACTER VOICE — {speaker_b}:\n{CHARACTER_VOICES[speaker_b]}"
    elif guest_profile and guest_profile.get("personality"):
        personality_block += f"\n\n{speaker_b}'s personality: {guest_profile['personality']}. Make their dialogue unmistakably reflect this."

    # Build angles focus
    angles_block = ""
    if angles:
        angles_block = "\n\nFOCUS ANGLES (emphasize these perspectives in the discussion):\n"
        angles_block += "\n".join(f"- {a}" for a in angles)

    style_prompts = {
        "executive": f"""Write a podcast script as a professional briefing between two senior analysts ({speaker_a} and {speaker_b}).
- Open with a concise hook about why this research matters NOW
- Discuss 3-4 key strategic findings with specific data points
- Include forward-looking analysis and implications
- Keep the tone sharp, analytical, and business-focused
- Close with actionable takeaways""",

        "curious": f"""Write a podcast script as an engaging interview between an enthusiastic {speaker_a} and a knowledgeable {speaker_b}.
- Open with {speaker_a} setting up the topic in an accessible way
- {speaker_a} asks the questions that a curious listener would want answered
- {speaker_b} explains complex findings using analogies and examples
- Include "aha moments" where surprising findings are revealed
- Close with the most important thing listeners should remember""",

        "debate": f"""Write a podcast script as an intellectual debate between two experts ({speaker_a} and {speaker_b}).
- Open by framing the central tension or controversy in the research
- Each expert advocates for different interpretations of the evidence
- Include respectful challenges: "But what about..." and "I'd push back on that..."
- Explore nuances and gray areas rather than declaring a winner
- Close with areas of agreement and remaining open questions""",
    }

    prompt = f"""You are a professional podcast scriptwriter. Write a natural, engaging podcast script.

RESEARCH QUERY: {query}

RESEARCH CONTENT:
{content}

STYLE INSTRUCTIONS:
{style_prompts.get(style, style_prompts["curious"])}{personality_block}{angles_block}

FORMAT RULES:
- Write as natural spoken dialogue — contractions, conversational flow, occasional humor
- Use speaker labels like "{speaker_a}:" and "{speaker_b}:" at the start of each turn
- Aim for 2000-4000 words (5-7 minutes when spoken)
- Reference specific findings, numbers, and sources from the research
- Make it sound like a real conversation, not a script being read
- CRITICAL: Each speaker must sound unmistakably like their character. A reader should be able to tell who's talking WITHOUT the speaker labels. Use their specific speech patterns, vocabulary, and mannerisms described in the CHARACTER VOICE sections above. Do NOT make them sound generic or interchangeable.

AUDIO TAGS (ElevenLabs v3):
This script will be read aloud by an AI TTS engine that supports audio tags in [square brackets]. USE THEM liberally to make the podcast sound alive and expressive. Place tags inline BEFORE the text they modify.

Emotion/delivery tags: [excited], [serious], [deadpan], [playfully], [nervously], [thoughtfully], [confidently], [skeptically], [warmly], [intensely]
Reaction tags: [laughs], [sighs], [gasps], [clears throat], [chuckles], [scoffs]
Delivery style tags: [whispers], [emphatically], [slowly], [quickly], [softly]

Example usage:
  {speaker_a}: [excited] Oh, this is the part I've been waiting to get to. [laughs] The numbers are absolutely wild. [serious] But here's what most people miss...
  {speaker_b}: [skeptically] Hold on, let me push back on that. [thoughtfully] If you look at the data from a different angle... [sighs] it tells a very different story.

Use 2-4 audio tags per speaker turn on average. Match the tags to each character's personality — e.g., a deadpan character uses [sighs] and [deadpan], an enthusiastic character uses [excited] and [gasps]. Do NOT overuse them — they should feel natural, not every sentence.

Write the script now:"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            config=GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=8000,
            ),
            contents=prompt,
        )
        script = resp.text or ""
        logger.info("Podcast script generated: style=%s, host=%s, guest=%s, angles=%d, length=%d chars",
                     style, speaker_a, speaker_b, len(angles or []), len(script))
        return script
    except Exception:
        logger.exception("Podcast script generation failed")
        raise
