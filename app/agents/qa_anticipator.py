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


def build_qa_anticipator(
    model: str = "gemini-2.5-flash",
    business_context: dict | None = None,
) -> LlmAgent:
    instruction = QA_ANTICIPATOR_INSTRUCTION

    if business_context:
        user_role = business_context.get("user_role", "")
        industry = business_context.get("industry", "")
        decision_type = business_context.get("decision_type", "")
        stakeholders = business_context.get("stakeholders", "")

        if any([user_role, industry, decision_type, stakeholders]):
            profile_parts = []
            if user_role:
                profile_parts.append(f"- Role: {user_role}")
            if industry:
                profile_parts.append(f"- Industry: {industry}")
            if decision_type:
                profile_parts.append(f"- Decision type: {decision_type}")
            if stakeholders:
                profile_parts.append(f"- Key stakeholders: {stakeholders}")

            instruction += (
                "\n\nThe reader has this profile:\n"
                + "\n".join(profile_parts)
                + "\n\nTailor questions to this person's perspective. "
                "Include questions their stakeholders would ask."
            )

    return LlmAgent(
        name="qa_anticipator",
        model=model,
        instruction=instruction,
        output_key="qa_clusters_raw",
    )
