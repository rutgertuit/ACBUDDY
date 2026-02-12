from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptTurn:
    role: str = ""
    message: str = ""
    time_in_call_secs: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptTurn":
        return cls(
            role=data.get("role", ""),
            message=data.get("message", ""),
            time_in_call_secs=float(data.get("time_in_call_secs", 0.0)),
        )


@dataclass
class WebhookPayload:
    event_type: str = ""
    conversation_id: str = ""
    agent_id: str = ""
    status: str = ""
    transcript: list[TranscriptTurn] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "WebhookPayload":
        event_type = data.get("type", "")
        inner = data.get("data", {})
        transcript_raw = inner.get("transcript", [])
        return cls(
            event_type=event_type,
            conversation_id=inner.get("conversation_id", ""),
            agent_id=inner.get("agent_id", ""),
            status=inner.get("status", ""),
            transcript=[TranscriptTurn.from_dict(t) for t in transcript_raw],
        )

    def extract_user_messages(self) -> str:
        """Concatenate all user turns into a single string."""
        return " ".join(
            turn.message for turn in self.transcript if turn.role == "user"
        )
