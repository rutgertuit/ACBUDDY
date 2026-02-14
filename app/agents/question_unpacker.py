from google.adk.agents import LlmAgent

UNPACKER_INSTRUCTION = """You are a research question decomposer.

Given a user's research query or conversation context, break it down into 2-5 specific,
searchable sub-questions that together will comprehensively address the user's needs.

Output ONLY a valid JSON array of strings. No explanation, no markdown, just the JSON array.

Example output:
["What are the current market trends for X?", "Who are the key players in X?", "What are recent innovations in X?"]
"""


def build_question_unpacker(model: str = "gemini-2.5-flash") -> LlmAgent:
    """Build an LlmAgent that decomposes a query into sub-questions."""
    return LlmAgent(
        name="question_unpacker",
        model=model,
        instruction=UNPACKER_INSTRUCTION,
        output_key="unpacked_questions",
    )
