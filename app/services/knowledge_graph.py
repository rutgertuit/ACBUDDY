"""Knowledge graph storage and operations — stores entities and relationships in GCS."""

import json
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

GRAPH_BLOB = "graph/knowledge_graph.json"


@dataclass
class KGEntity:
    name: str
    type: str  # company, person, product, concept, technology, regulation, market
    aliases: list[str] = field(default_factory=list)
    source_jobs: list[str] = field(default_factory=list)  # job_ids that mentioned this entity


@dataclass
class KGRelationship:
    from_entity: str
    to_entity: str
    type: str  # competes_with, produces, regulates, etc.
    description: str = ""
    source_jobs: list[str] = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    entities: dict[str, KGEntity] = field(default_factory=dict)  # keyed by normalized name
    relationships: list[KGRelationship] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "relationships": [asdict(r) for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        graph = cls()
        for name, edata in data.get("entities", {}).items():
            graph.entities[name] = KGEntity(**edata)
        for rdata in data.get("relationships", []):
            graph.relationships.append(KGRelationship(**rdata))
        return graph


def _normalize_name(name: str) -> str:
    """Normalize an entity name for consistent lookup."""
    return name.strip().lower()


def load_graph(bucket_name: str) -> KnowledgeGraph:
    """Load the knowledge graph from GCS, or return empty graph."""
    if not bucket_name:
        return KnowledgeGraph()
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(GRAPH_BLOB)
        if not blob.exists():
            return KnowledgeGraph()
        data = json.loads(blob.download_as_text())
        return KnowledgeGraph.from_dict(data)
    except Exception:
        logger.exception("Failed to load knowledge graph")
        return KnowledgeGraph()


def save_graph(graph: KnowledgeGraph, bucket_name: str) -> None:
    """Save the knowledge graph to GCS."""
    if not bucket_name:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(GRAPH_BLOB)
        blob.upload_from_string(
            json.dumps(graph.to_dict(), indent=2),
            content_type="application/json",
        )
        logger.info("Saved knowledge graph: %d entities, %d relationships",
                     len(graph.entities), len(graph.relationships))
    except Exception:
        logger.exception("Failed to save knowledge graph")


def merge_extraction(graph: KnowledgeGraph, extraction: dict, job_id: str) -> None:
    """Merge extracted entities and relationships into the graph.

    Args:
        graph: Existing knowledge graph (modified in-place).
        extraction: Dict with "entities" and "relationships" from entity_extractor.
        job_id: Source job ID for provenance.
    """
    # Merge entities
    for entity in extraction.get("entities", []):
        name = entity.get("name", "")
        if not name:
            continue
        key = _normalize_name(name)

        if key in graph.entities:
            existing = graph.entities[key]
            # Merge aliases
            new_aliases = set(existing.aliases) | set(entity.get("aliases", []))
            existing.aliases = list(new_aliases)
            # Add source job
            if job_id not in existing.source_jobs:
                existing.source_jobs.append(job_id)
        else:
            graph.entities[key] = KGEntity(
                name=name,
                type=entity.get("type", "concept"),
                aliases=entity.get("aliases", []),
                source_jobs=[job_id],
            )

    # Merge relationships
    existing_rels = {
        (_normalize_name(r.from_entity), _normalize_name(r.to_entity), r.type)
        for r in graph.relationships
    }

    for rel in extraction.get("relationships", []):
        from_name = rel.get("from", "")
        to_name = rel.get("to", "")
        rel_type = rel.get("type", "")
        if not from_name or not to_name or not rel_type:
            continue

        key = (_normalize_name(from_name), _normalize_name(to_name), rel_type)
        if key in existing_rels:
            # Update source_jobs for existing relationship
            for existing_rel in graph.relationships:
                if (_normalize_name(existing_rel.from_entity),
                    _normalize_name(existing_rel.to_entity),
                    existing_rel.type) == key:
                    if job_id not in existing_rel.source_jobs:
                        existing_rel.source_jobs.append(job_id)
                    break
        else:
            graph.relationships.append(KGRelationship(
                from_entity=from_name,
                to_entity=to_name,
                type=rel_type,
                description=rel.get("description", ""),
                source_jobs=[job_id],
            ))
            existing_rels.add(key)


def find_connections(graph: KnowledgeGraph, entity_name: str) -> dict:
    """Find all connections for a given entity.

    Returns dict with entity info and lists of outgoing/incoming relationships.
    """
    key = _normalize_name(entity_name)
    entity = graph.entities.get(key)
    if not entity:
        # Check aliases
        for k, e in graph.entities.items():
            if entity_name.lower() in [a.lower() for a in e.aliases]:
                entity = e
                key = k
                break

    if not entity:
        return {"found": False, "entity_name": entity_name}

    outgoing = []
    incoming = []
    for rel in graph.relationships:
        if _normalize_name(rel.from_entity) == key:
            outgoing.append({
                "to": rel.to_entity, "type": rel.type,
                "description": rel.description, "sources": len(rel.source_jobs),
            })
        if _normalize_name(rel.to_entity) == key:
            incoming.append({
                "from": rel.from_entity, "type": rel.type,
                "description": rel.description, "sources": len(rel.source_jobs),
            })

    return {
        "found": True,
        "entity": asdict(entity),
        "outgoing": outgoing,
        "incoming": incoming,
    }


def get_graph_stats(graph: KnowledgeGraph) -> dict:
    """Get summary statistics of the knowledge graph."""
    type_counts = {}
    for e in graph.entities.values():
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    rel_type_counts = {}
    for r in graph.relationships:
        rel_type_counts[r.type] = rel_type_counts.get(r.type, 0) + 1

    return {
        "total_entities": len(graph.entities),
        "total_relationships": len(graph.relationships),
        "entity_types": type_counts,
        "relationship_types": rel_type_counts,
    }
