```js
display(html`<nav class="acb-nav">
  <a href="/" class="acb-nav-back">&larr; Luminary</a>
  <span class="acb-nav-sep">|</span>
  <a href="/explore/">Archive</a>
  <a href="/explore/graph.html" class="acb-nav-active">Knowledge Graph</a>
  <a href="/explore/performance.html">Performance</a>
</nav>`);
```

# Knowledge Graph Explorer

```js
import {getGraph, getEntityConnections} from "./components/api.js";
import {ForceGraph} from "./components/force-graph.js";

const graphData = await getGraph();
const entities = graphData.entities ?? [];
const relationships = graphData.relationships ?? [];
const graphStats = graphData.stats ?? {};

const entityTypes = [...new Set(entities.map(e => e.type))].sort();
```

```js
display(html`<div class="card-row">
  <div class="card">
    <div class="label">Entities</div>
    <div class="big">${graphStats.total_entities ?? entities.length}</div>
  </div>
  <div class="card">
    <div class="label">Relationships</div>
    <div class="big">${graphStats.total_relationships ?? relationships.length}</div>
  </div>
  <div class="card">
    <div class="label">Entity Types</div>
    <div class="big">${entityTypes.length}</div>
  </div>
</div>`);
```

## Filters

```js
const entitySearch = view(Inputs.text({placeholder: "Search entities...", width: 260}));
const typeFilter = view(Inputs.select(["All", ...entityTypes], {value: "All", label: "Type"}));
const minMentions = view(Inputs.range([1, Math.max(10, ...entities.map(e => e.mentions ?? 1))], {
  value: 1, step: 1, label: "Min mentions"
}));
```

```js
const filteredEntities = entities.filter(e => {
  if (entitySearch && !e.name.toLowerCase().includes(entitySearch.toLowerCase())) return false;
  if (typeFilter !== "All" && e.type !== typeFilter) return false;
  if ((e.mentions ?? 1) < minMentions) return false;
  return true;
});

const filteredNames = new Set(filteredEntities.map(e => e.name));
const filteredRels = relationships.filter(r =>
  filteredNames.has(r.from) && filteredNames.has(r.to)
);
```

## Graph

```js
const selectedEntity = Mutable(null);

if (filteredEntities.length > 0) {
  const graphEl = ForceGraph(filteredEntities, filteredRels, {
    width: Math.min(900, window.innerWidth - 80),
    height: 500,
    onNodeClick: (name) => selectedEntity.value = name
  });
  display(graphEl);
} else {
  display(html`<div class="empty-state">No entities match the current filters.</div>`);
}
```

## Entity Detail

```js
const selected = selectedEntity;
```

```js
if (selected) {
  const detail = await getEntityConnections(selected);
  if (detail && detail.found) {
    const e = detail.entity;
    const outgoing = detail.outgoing ?? [];
    const incoming = detail.incoming ?? [];

    const outRows = outgoing.map(r => `<tr><td>${r.type}</td><td>${r.to}</td><td style="color:var(--acb-text-muted)">${r.description ?? ""}</td></tr>`).join("");
    const inRows = incoming.map(r => `<tr><td>${r.from}</td><td>${r.type}</td><td style="color:var(--acb-text-muted)">${r.description ?? ""}</td></tr>`).join("");

    display(html`<div class="detail-panel">
      <h3>${e.name} <span class="entity-type">${e.type}</span></h3>
      ${e.aliases?.length ? html`<p style="color:var(--acb-text-muted);font-size:0.85rem;">Aliases: ${e.aliases.join(", ")}</p>` : ""}
      ${outgoing.length > 0 ? html`
        <h4 style="margin-bottom:0.5rem;">Outgoing (${outgoing.length})</h4>
        <table class="data-table">
          <thead><tr><th>Relationship</th><th>Target</th><th>Description</th></tr></thead>
          <tbody>${outRows}</tbody>
        </table>` : ""}
      ${incoming.length > 0 ? html`
        <h4 style="margin-bottom:0.5rem;">Incoming (${incoming.length})</h4>
        <table class="data-table">
          <thead><tr><th>Source</th><th>Relationship</th><th>Description</th></tr></thead>
          <tbody>${inRows}</tbody>
        </table>` : ""}
    </div>`);
  } else {
    display(html`<div class="detail-panel"><p>Entity not found.</p></div>`);
  }
} else {
  display(html`<div class="empty-state">Click a node in the graph to see its connections.</div>`);
}
```
