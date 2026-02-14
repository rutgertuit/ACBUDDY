from dataclasses import dataclass, field


@dataclass
class StudyResult:
    title: str = ""
    angle: str = ""
    questions: list[str] = field(default_factory=list)
    rounds: list[dict[str, str]] = field(default_factory=list)
    synthesis: str = ""
    doc_id: str = ""


@dataclass
class QAClusterResult:
    theme: str = ""
    questions: list[str] = field(default_factory=list)
    findings: str = ""
    doc_id: str = ""


@dataclass
class ResearchResult:
    original_query: str = ""

    # STANDARD / QUICK fields (backward compatible)
    unpacked_questions: list[str] = field(default_factory=list)
    research_findings: dict[str, str] = field(default_factory=dict)
    follow_up_questions: list[str] = field(default_factory=list)
    follow_up_findings: dict[str, str] = field(default_factory=dict)
    final_synthesis: str = ""

    # DEEP fields
    study_plan: list[dict] = field(default_factory=list)
    studies: list[StudyResult] = field(default_factory=list)
    master_synthesis: str = ""
    master_doc_id: str = ""
    qa_clusters: list[QAClusterResult] = field(default_factory=list)
    qa_summary: str = ""
    qa_summary_doc_id: str = ""
    all_doc_ids: list[str] = field(default_factory=list)

    # Strategic analysis (populated by strategic analyst)
    strategic_analysis: str = ""

    # Query analysis (populated by query_analyzer in Phase 0)
    query_analysis: dict = field(default_factory=dict)

    # Claim validation (populated by claim_validator after synthesis)
    claim_validation: dict = field(default_factory=dict)

    # Synthesis quality (populated by synthesis evaluator)
    synthesis_score: float = 0.0
    synthesis_scores: dict = field(default_factory=dict)
    refinement_rounds: int = 0
