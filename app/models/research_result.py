from dataclasses import dataclass, field


@dataclass
class ResearchResult:
    original_query: str = ""
    unpacked_questions: list[str] = field(default_factory=list)
    research_findings: dict[str, str] = field(default_factory=dict)
    follow_up_questions: list[str] = field(default_factory=list)
    follow_up_findings: dict[str, str] = field(default_factory=dict)
    final_synthesis: str = ""
