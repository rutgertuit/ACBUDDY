import io
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elevenlabs.io/v1"


def _headers(api_key: str) -> dict:
    return {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }


def get_conversation(conversation_id: str, api_key: str) -> dict:
    """Fetch full conversation data from ElevenLabs."""
    url = f"{BASE_URL}/convai/conversations/{conversation_id}"
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    return resp.json()


def format_conversation_context(data: dict) -> str:
    """Format conversation data into a human-readable transcript."""
    lines = []
    transcript = data.get("transcript", [])
    for turn in transcript:
        role = turn.get("role", "unknown").capitalize()
        message = turn.get("message", "")
        lines.append(f"{role}: {message}")
    return "\n".join(lines)


def upload_to_knowledge_base(text: str, name: str, api_key: str) -> str:
    """Upload text as a document to ElevenLabs Knowledge Base. Returns document ID."""
    url = f"{BASE_URL}/convai/knowledge-base/text"
    payload = {
        "name": name,
        "text": text,
    }
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    doc_id = result.get("id", result.get("document_id", ""))
    logger.info("Uploaded KB document: %s (id=%s)", name, doc_id)
    return doc_id


def attach_document_to_agent(agent_id: str, doc_id: str, doc_name: str, api_key: str) -> None:
    """Attach a KB document to an agent using GET-then-PATCH to preserve existing docs."""
    headers = _headers(api_key)

    # GET current agent config
    get_url = f"{BASE_URL}/convai/agents/{agent_id}"
    resp = requests.get(get_url, headers=headers, timeout=30)
    resp.raise_for_status()
    agent_config = resp.json()

    # Extract existing knowledge base docs
    convai_config = agent_config.get("conversation_config", {})
    agent_section = convai_config.get("agent", {})
    prompt_section = agent_section.get("prompt", {})
    existing_kb = prompt_section.get("knowledge_base", [])

    # Check if document is already attached
    existing_ids = {doc.get("id", doc.get("document_id", "")) for doc in existing_kb}
    if doc_id in existing_ids:
        logger.info("Document %s already attached to agent %s", doc_id, agent_id)
        return

    # Append new document
    existing_kb.append({"type": "text", "id": doc_id, "name": doc_name})

    # PATCH agent with updated knowledge base
    patch_url = f"{BASE_URL}/convai/agents/{agent_id}"
    patch_payload = {
        "conversation_config": {
            "agent": {
                "prompt": {
                    "knowledge_base": existing_kb,
                }
            }
        }
    }
    resp = requests.patch(patch_url, headers=headers, json=patch_payload, timeout=30)
    resp.raise_for_status()
    logger.info("Attached document %s to agent %s", doc_id, agent_id)


def list_agent_knowledge_base(agent_id: str, api_key: str) -> list[dict]:
    """Return the knowledge_base array from an agent's config."""
    url = f"{BASE_URL}/convai/agents/{agent_id}"
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    agent_config = resp.json()
    kb = (
        agent_config
        .get("conversation_config", {})
        .get("agent", {})
        .get("prompt", {})
        .get("knowledge_base", [])
    )
    return kb


def detach_document_from_agent(agent_id: str, doc_id: str, api_key: str) -> None:
    """Remove a KB document from an agent (GET current list, filter, PATCH back)."""
    headers = _headers(api_key)
    url = f"{BASE_URL}/convai/agents/{agent_id}"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    agent_config = resp.json()

    existing_kb = (
        agent_config
        .get("conversation_config", {})
        .get("agent", {})
        .get("prompt", {})
        .get("knowledge_base", [])
    )

    filtered = [d for d in existing_kb if d.get("id", d.get("document_id", "")) != doc_id]
    if len(filtered) == len(existing_kb):
        logger.info("Document %s not found on agent %s, nothing to detach", doc_id, agent_id)
        return

    patch_payload = {
        "conversation_config": {
            "agent": {
                "prompt": {
                    "knowledge_base": filtered,
                }
            }
        }
    }
    resp = requests.patch(url, headers=headers, json=patch_payload, timeout=30)
    resp.raise_for_status()
    logger.info("Detached document %s from agent %s", doc_id, agent_id)


def attach_documents_to_agent(agent_id: str, doc_map: dict[str, str], api_key: str) -> None:
    """Attach multiple KB documents to an agent in a single GET + PATCH.

    Args:
        agent_id: ElevenLabs agent ID.
        doc_map: Mapping of document ID to document name.
        api_key: ElevenLabs API key.
    """
    if not doc_map:
        return

    headers = _headers(api_key)

    get_url = f"{BASE_URL}/convai/agents/{agent_id}"
    resp = requests.get(get_url, headers=headers, timeout=30)
    resp.raise_for_status()
    agent_config = resp.json()

    convai_config = agent_config.get("conversation_config", {})
    agent_section = convai_config.get("agent", {})
    prompt_section = agent_section.get("prompt", {})
    existing_kb = prompt_section.get("knowledge_base", [])

    existing_ids = {doc.get("id", doc.get("document_id", "")) for doc in existing_kb}
    new_docs = [{"type": "text", "id": did, "name": dname} for did, dname in doc_map.items() if did not in existing_ids]

    if not new_docs:
        logger.info("All %d documents already attached to agent %s", len(doc_map), agent_id)
        return

    existing_kb.extend(new_docs)

    patch_url = f"{BASE_URL}/convai/agents/{agent_id}"
    patch_payload = {
        "conversation_config": {
            "agent": {
                "prompt": {
                    "knowledge_base": existing_kb,
                }
            }
        }
    }
    resp = requests.patch(patch_url, headers=headers, json=patch_payload, timeout=30)
    resp.raise_for_status()
    logger.info("Attached %d new documents to agent %s (total KB: %d)", len(new_docs), agent_id, len(existing_kb))
