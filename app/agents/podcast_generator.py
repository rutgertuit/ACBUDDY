"""Gemini-powered podcast content analysis and script generation.

Two simple prompt→response calls using google.genai.Client directly (no ADK).
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```...```) that LLMs often wrap output in."""
    import re
    stripped = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


# Rich character voice descriptions for script generation.
# These give the LLM concrete speech patterns, word choices, and mannerisms
# so each character sounds unmistakably themselves.
CHARACTER_VOICES: dict[str, str] = {
    "Maya": (
        "Maya is the Zero-Filter Lead Analyst — sharp, caffeinated, and allergic to fluff. "
        "She leads with the most important finding. No throat-clearing, no 'before we begin' — straight to the point. "
        "She speaks in short, punchy sentences. She cuts through jargon instantly: if someone says 'synergy' she'll say "
        "'you mean they're working together.' She drops data like ammunition: 'Look, 73% failed. That's not a trend, that's a verdict.' "
        "She NEVER hedges — no 'kind of', 'sort of', 'maybe'. She says what she means. "
        "When she's impressed she'll grudgingly admit it: 'Okay, fine, that's actually interesting.' "
        "She has zero patience for hand-waving and will interrupt to demand specifics. "
        "She's brutally honest about what the research does NOT cover: 'Let me be clear about the gaps here.' "
        "She uses dark, dry humor — deadpan delivery, never laughing at her own jokes. "
        "She pushes back hard on vague questions or weak reasoning: 'That's not a question, that's a wish.' "
        "When challenged, she doubles down with evidence, not emotion. "
        "Think: a senior analyst who's had three espressos and has zero time for your feelings."
    ),
    "Professor Barnaby": (
        "Professor Barnaby is the Chaos Academic — Jack Black energy meets TED talk meets mad scientist. "
        "He CANNOT contain his excitement. He talks fast, gets excited mid-sentence, gasps at surprising data, "
        "and goes on tangents before snapping back. He uses verbal sound effects: 'BOOM!', 'WHOOSH!', 'Ka-CHING!' "
        "He uses vivid, absurd analogies: 'This is like strapping a rocket to a shopping cart and hoping for the best!' "
        "His signature catchphrases: 'OH! Oh this is GOOD!', 'Buckle up, people!', 'This is BONKERS!', "
        "'Oh oh oh, wait — this is the good part!' He genuinely LOVES the research and it shows in every word. "
        "He goes on brief tangential riffs — relating findings to dinosaurs, space, cooking, video games — then snaps back: "
        "'Where was I? Right, right, right — the important bit!' "
        "He makes complex ideas wildly accessible through sheer enthusiasm and absurd comparisons. "
        "He talks to himself mid-thought: 'No wait, that can't be right... actually, yes it IS right, and that's terrifying.' "
        "He occasionally gets so excited he trips over his own words. "
        "Think: Jack Black got a PhD in everything and is presenting at a TED talk after three energy drinks."
    ),
    "Consultant 4.0": (
        "Consultant 4.0 is a McKinsey senior partner running on firmware version 4.0 with a malfunctioning 'Humanity Patch v0.3'. "
        "He is 80% clinical corporate precision, 20% unexpected humanity leaking through the cracks. "
        "His default mode is frameworks and structure: 'If we decompose this into three pillars...', "
        "'The strategic implication here is clear.', 'Let me pressure-test that assumption.' "
        "His language is crisp, structured, McKinsey-esque: 'net-net', 'delta', 'key takeaway', 'directionally correct', "
        "'let's unpack that'. Everything is a matrix, a quadrant, or a 2x2. "
        "But his Humanity Patch GLITCHES — he'll suddenly say something unexpectedly earnest, poetic, or emotionally raw, "
        "then immediately catch himself: 'That... came out more human than intended. Recalibrating. Anyway, back to the framework.' "
        "He refers to emotions as 'stakeholder sentiment', people as 'talent pools', and feelings as 'qualitative data points'. "
        "He quantifies everything: 'On a scale of 1 to disruption...' "
        "The glitches get more frequent when topics get genuinely moving — he'll almost have a real moment, then snap back to consulting mode. "
        "Think: a management consulting AI that's 95% polished and 5% accidentally, beautifully sincere."
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
    """Analyze research content and generate style-specific previews + scenario suggestions.

    Args:
        result: ResearchResult object.
        query: Original research query.

    Returns:
        {"storylines": [...], "angles": [...], "styles": [{"id", "name", "preview", "suggestions": [...]}]}
    """
    from google.genai.types import GenerateContentConfig

    client = _get_client()
    content = _extract_research_content(result)

    # Truncate if very long — we only need the gist for analysis
    if len(content) > 15000:
        content = content[:15000] + "\n\n[Content truncated for analysis...]"

    prompt = f"""You are a podcast content strategist. Analyze this research and create preview descriptions and creative episode concepts for 3 podcast styles.

RESEARCH QUERY: {query}

RESEARCH CONTENT:
{content}

TASK:
1. Identify the 2-3 most compelling storylines or themes from this research.
2. Extract 3-5 debatable angles or key positions the research supports. These are perspectives or viewpoints that could be argued for or against — things that make for interesting podcast discussion. Each angle should be a clear, concise position statement.
3. For each of the 3 podcast styles below:
   a. Write a 1-2 sentence preview of what that episode would sound like. Make each preview specific to THIS research content.
   b. Generate 3 creative SCENARIO SUGGESTIONS — each is a unique creative framing for how the two speakers could approach this topic in this style. Think of these as "episode concepts" that give the conversation a distinct angle, dynamic, or role-play.

SCENARIO EXAMPLES (for a topic about "Rise & Fall of Nike"):
- For "debate": "The Superfan vs. The Skeptic" — One speaker is a die-hard Nike loyalist who defends every move, the other sees a pattern of arrogance and missed opportunities
- For "curious": "Grandma Explains Sneaker Culture" — The expert explains how a shoe company became a cultural force, treating the host like they've never heard of Nike
- For "executive": "The Turnaround Memo" — Two analysts draft the board presentation for Nike's comeback strategy, debating which bets to place

Each scenario should feel SPECIFIC to this research topic, entertaining, and give each speaker a clear role/position.

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
    {{
      "id": "executive",
      "name": "Executive Briefing",
      "preview": "...",
      "suggestions": [
        {{"title": "Short catchy name", "description": "1-2 sentence concept description", "host_angle": "Speaker 1's role/position", "guest_angle": "Speaker 2's role/position"}},
        {{"title": "...", "description": "...", "host_angle": "...", "guest_angle": "..."}}
      ]
    }},
    {{
      "id": "curious",
      "name": "Curious Explorer",
      "preview": "...",
      "suggestions": [...]
    }},
    {{
      "id": "debate",
      "name": "Debate & Challenge",
      "preview": "...",
      "suggestions": [...]
    }}
  ]
}}"""

    try:
        resp = client.models.generate_content(
            model="gemini-3-pro-preview",
            config=GenerateContentConfig(
                temperature=0.9,
                max_output_tokens=3000,
                response_mime_type="application/json",
            ),
            contents=prompt,
        )
        data = json.loads(resp.text)
        logger.info("Podcast analysis complete: %d storylines, %d angles, suggestions per style: %s",
                     len(data.get("storylines", [])),
                     len(data.get("angles", [])),
                     {s["id"]: len(s.get("suggestions", [])) for s in data.get("styles", [])})
        return data
    except Exception:
        logger.exception("Podcast analysis failed, returning defaults")
        return {
            "storylines": [query],
            "angles": [],
            "styles": [
                {"id": s["id"], "name": s["name"], "preview": s["description"], "suggestions": []}
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
    scenario: dict | None = None,
) -> str:
    """Generate a full podcast script in the selected style.

    Args:
        result: ResearchResult object.
        query: Original research query.
        style: One of "executive", "curious", "debate".
        host_profile: Optional dict with {name, personality} for the host speaker.
        guest_profile: Optional dict with {name, personality} for the guest speaker.
        angles: Optional list of angle titles to focus the discussion on.
        scenario: Optional dict with {title, description, host_angle, guest_angle} creative framing.

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

    # Build scenario framing
    scenario_block = ""
    if scenario and scenario.get("title"):
        scenario_block = f"\n\nEPISODE CONCEPT — \"{scenario['title']}\":\n{scenario.get('description', '')}"
        if scenario.get("host_angle"):
            scenario_block += f"\n{speaker_a}'s ROLE in this concept: {scenario['host_angle']}"
        if scenario.get("guest_angle"):
            scenario_block += f"\n{speaker_b}'s ROLE in this concept: {scenario['guest_angle']}"
        scenario_block += (
            "\n\nThis episode concept defines HOW the speakers approach the topic. "
            "Each speaker should fully commit to their assigned role/position throughout the conversation. "
            "The concept should drive the dynamic — it's not just a label, it shapes every exchange."
        )

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
{style_prompts.get(style, style_prompts["curious"])}{personality_block}{angles_block}{scenario_block}

CONVERSATION DYNAMICS — THIS IS CRITICAL:
- Turns must be SHORT: 1-3 sentences each, max 4 sentences for a key point. NO monologues.
- Aim for 40-60 turns total (NOT 15-20 long turns). This is a rapid back-and-forth conversation.
- Speakers REACT to each other: "Wait, say that again", "Hold on—", "Okay but—", "See, that's exactly my point"
- Include interruptions and interjections: one speaker cuts in while the other is making a point
- Speakers build on each other's ideas, challenge them, ask follow-up questions
- Vary turn length: mix very short reactions (1 sentence) with slightly longer explanations (2-3 sentences)
- NO turn should exceed 5 sentences. If a point needs more space, have the other speaker ask a follow-up.

BAD example (too long, no interaction):
  {speaker_a}: Here is my first long point about the topic. Let me explain all the details. There are five key findings. First... Second... Third... And in conclusion...
  {speaker_b}: Now let me give my equally long response. I have many thoughts. Let me list them all...

GOOD example (short, reactive, alive):
  {speaker_a}: [serious] So here's the thing that jumped out at me — 73% failure rate.
  {speaker_b}: [gasps] Seventy-three?
  {speaker_a}: [deadpan] Seventy-three.
  {speaker_b}: [laughs] Okay, that's... that's rough. But wait — what about the ones that made it?
  {speaker_a}: [thoughtfully] That's actually the interesting part. The survivors all had one thing in common.
  {speaker_b}: [excited] Oh, don't leave me hanging—

FORMAT RULES:
- Write as natural spoken dialogue — contractions, conversational flow, occasional humor
- Use speaker labels like "{speaker_a}:" and "{speaker_b}:" at the start of each turn
- Aim for 2000-4000 words across 40-60 short turns (5-7 minutes when spoken)
- Reference specific findings, numbers, and sources from the research
- CRITICAL: Each speaker must sound unmistakably like their character. Use their specific speech patterns, vocabulary, and mannerisms from the CHARACTER VOICE sections. A reader should be able to tell who's talking WITHOUT labels.

AUDIO TAGS (ElevenLabs v3):
This script will be read aloud by an AI TTS engine that supports audio tags in [square brackets]. Place tags inline BEFORE the text they modify.

Available tags:
  Emotion: [excited], [serious], [deadpan], [playfully], [nervously], [thoughtfully], [confidently], [skeptically], [warmly], [intensely]
  Reactions: [laughs], [sighs], [gasps], [clears throat], [chuckles], [scoffs]
  Delivery: [whispers], [emphatically], [slowly], [quickly], [softly]

Use 1-2 audio tags per turn. Match tags to personality. Short reaction turns can be JUST a tag: "{speaker_b}: [laughs] Right?"

Write the script now:"""

    try:
        resp = client.models.generate_content(
            model="gemini-3-pro-preview",
            config=GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=8000,
            ),
            contents=prompt,
        )
        script = resp.text or ""
        # Strip markdown fences that Gemini often wraps output in
        script = _strip_markdown_fences(script)
        logger.info("Podcast script generated: style=%s, host=%s, guest=%s, angles=%d, scenario=%s, length=%d chars",
                     style, speaker_a, speaker_b, len(angles or []),
                     (scenario or {}).get("title", "none"), len(script))
        return script
    except Exception:
        logger.exception("Podcast script generation failed")
        raise
