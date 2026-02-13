from google.adk.agents import LlmAgent

QA_ANTICIPATOR_INSTRUCTION = """You are a research anticipation specialist. Given a comprehensive
research briefing, generate likely follow-up questions that a reader would ask.

The research findings are:
{master_synthesis}

Generate 5-15 follow-up questions (scale with topic complexity) and group them into 3-5 thematic clusters.

Output ONLY valid JSON:
{
  "clusters": [
    {
      "theme": "Theme Name",
      "questions": ["Question 1?", "Question 2?", "Question 3?"]
    }
  ]
}

Focus on:
- Questions that go deeper into key findings
- Questions about implications and next steps
- Questions that challenge assumptions
- Practical "so what" questions

No explanation, no markdown fences, just the JSON."""


def build_qa_anticipator(model: str = "gemini-2.0-flash") -> LlmAgent:
    return LlmAgent(
        name="qa_anticipator",
        model=model,
        instruction=QA_ANTICIPATOR_INSTRUCTION,
        output_key="qa_clusters_raw",
    )
