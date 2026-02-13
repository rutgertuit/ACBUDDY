from google.adk.agents import LlmAgent

STUDY_PLANNER_INSTRUCTION = """You are a research study planner. Given a user's research query,
decompose it into 2-6 distinct research studies. Each study explores a different angle,
perspective, or dimension of the topic.

Consider the user's specific requests for angles, comparisons, or perspectives. If the user
mentions specific markets, stakeholders, or comparison axes, ensure each gets its own study.

Output ONLY a valid JSON array of objects. Each object must have:
- "title": A concise study title
- "angle": The perspective or focus area (1 sentence)
- "questions": An array of 2-4 specific, searchable research questions for this study

Example output:
[
  {
    "title": "Consumer Behavior & Leaflet Usage Patterns",
    "angle": "Understanding how consumers interact with and respond to leaflets",
    "questions": [
      "What percentage of consumers read retail leaflets?",
      "How do digital vs print leaflets compare in consumer engagement?",
      "What drives consumer response to leaflet promotions?"
    ]
  }
]

No explanation, no markdown fences, just the JSON array."""


def build_study_planner(model: str = "gemini-2.0-flash") -> LlmAgent:
    return LlmAgent(
        name="study_planner",
        model=model,
        instruction=STUDY_PLANNER_INSTRUCTION,
        output_key="study_plan",
    )
