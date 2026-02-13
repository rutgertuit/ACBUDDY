from enum import Enum

DEEP_KEYWORDS = ["deep dive", "comprehensive", "in-depth", "thorough analysis", "detailed research"]
QUICK_KEYWORDS = ["quick research", "brief", "quick look", "short summary", "fast research"]


class ResearchDepth(Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


def detect_depth(user_text: str) -> ResearchDepth:
    lower = user_text.lower()
    for kw in DEEP_KEYWORDS:
        if kw in lower:
            return ResearchDepth.DEEP
    for kw in QUICK_KEYWORDS:
        if kw in lower:
            return ResearchDepth.QUICK
    return ResearchDepth.STANDARD
