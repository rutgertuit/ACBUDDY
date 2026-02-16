```js
display(html`<nav class="acb-nav">
  <a href="/" class="acb-nav-back">&larr; Luminary</a>
  <span class="acb-nav-sep">|</span>
  <a href="/explore/" class="acb-nav-active">Archive</a>
  <a href="/explore/graph.html">Knowledge Graph</a>
  <a href="/explore/performance.html">Performance</a>
</nav>`);
```

# Archive Dashboard

```js
import {getArchive, getStats, formatDuration, scoreColor, badgeClass} from "./components/api.js";

const archive = await getArchive();
const stats = await getStats();

// Sort by date descending
const sorted = archive
  .map(d => ({...d, date: new Date(d.completed_at || d.created_at)}))
  .sort((a, b) => b.date - a.date);

// Compute aggregates
const deepRuns = sorted.filter(d => d.depth === "DEEP");
const avgDeepTime = deepRuns.length > 0
  ? deepRuns.reduce((s, d) => {
      const timings = d.phase_timings || {};
      const total = Object.values(timings).reduce((a, t) => a + (t.duration || 0), 0);
      return s + total;
    }, 0) / deepRuns.length
  : 0;
const scoresArr = deepRuns.map(d => d.synthesis_score).filter(s => s != null && s > 0);
const avgScore = scoresArr.length > 0
  ? Math.round(scoresArr.reduce((a, b) => a + b, 0) / scoresArr.length * 10) / 10
  : null;
```

```js
display(html`<div class="card-row">
  <div class="card">
    <div class="label">Completed</div>
    <div class="big">${stats.completed ?? sorted.length}</div>
  </div>
  <div class="card">
    <div class="label">Avg DEEP Duration</div>
    <div class="big">${formatDuration(avgDeepTime)}</div>
  </div>
  <div class="card">
    <div class="label">Avg Quality Score</div>
    <div class="big ${scoreColor(avgScore)}">${avgScore ?? "—"}<span class="unit">/ 10</span></div>
  </div>
  <div class="card">
    <div class="label">Currently Researching</div>
    <div class="big">${stats.researching ?? 0}</div>
  </div>
</div>`);
```

## Filters

```js
const searchText = view(Inputs.text({placeholder: "Search queries...", width: 300}));
const depthFilter = view(Inputs.select(["All", "QUICK", "STANDARD", "DEEP"], {value: "All", label: "Depth"}));
const minScore = view(Inputs.range([0, 10], {value: 0, step: 0.5, label: "Min quality score"}));
```

```js
const filtered = sorted.filter(d => {
  if (searchText && !d.query?.toLowerCase().includes(searchText.toLowerCase())) return false;
  if (depthFilter !== "All" && d.depth !== depthFilter) return false;
  if (minScore > 0 && (d.synthesis_score ?? 0) < minScore) return false;
  return true;
});
```

## Research Runs (${filtered.length})

```js
const tableHtml = (() => {
  if (filtered.length === 0) return `<div class="empty-state">No matching research runs found.</div>`;
  const rows = filtered.map(d => {
    const totalDur = d.phase_timings
      ? Object.values(d.phase_timings).reduce((a, t) => a + (t.duration || 0), 0)
      : 0;
    const score = d.synthesis_score;
    const sc = scoreColor(score);
    const dateStr = d.date.toLocaleDateString("en-GB", {day: "numeric", month: "short", year: "numeric"});
    return `<tr>
      <td><a href="/explore/research/index.html?id=${d.job_id}">${d.query?.slice(0, 80) || "—"}</a></td>
      <td><span class="${badgeClass(d.depth)}">${d.depth}</span></td>
      <td class="${sc}">${score != null ? score.toFixed(1) : "—"}</td>
      <td>${d.num_studies ?? "—"}</td>
      <td>${formatDuration(totalDur)}</td>
      <td>${dateStr}</td>
    </tr>`;
  }).join("");
  return `<table class="data-table">
    <thead><tr><th>Query</th><th>Depth</th><th>Score</th><th>Studies</th><th>Duration</th><th>Date</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
})();
display(html`${tableHtml}`);
```

## Research Volume (Weekly)

```js
const weeklyData = sorted.map(d => ({date: d.date, depth: d.depth}));
display(Plot.plot({
  width: Math.min(900, window.innerWidth - 80),
  height: 200,
  marginLeft: 40,
  x: {label: null},
  y: {label: "Runs"},
  color: {domain: ["QUICK", "STANDARD", "DEEP"], range: ["#3b82f6", "#eab308", "#8b5cf6"]},
  marks: [
    Plot.rectY(weeklyData, Plot.binX({y: "count"}, {x: "date", fill: "depth", interval: "week", tip: true})),
    Plot.ruleY([0])
  ]
}));
```

## Quality Score Distribution (DEEP)

```js
if (scoresArr.length > 0) {
  display(Plot.plot({
    width: Math.min(600, window.innerWidth - 80),
    height: 200,
    marginLeft: 40,
    x: {label: "Quality Score", domain: [0, 10]},
    y: {label: "Count"},
    marks: [
      Plot.rectY(scoresArr, Plot.binX({y: "count"}, {x: d => d, thresholds: 20, fill: "#0dccf2", tip: true})),
      Plot.ruleY([0]),
      Plot.ruleX([7], {stroke: "#22c55e", strokeDasharray: "4,4", strokeWidth: 2})
    ]
  }));
} else {
  display(html`<div class="empty-state">No DEEP runs with quality scores yet.</div>`);
}
```

## Depth Breakdown

```js
const depthCounts = ["QUICK", "STANDARD", "DEEP"].map(d => ({
  depth: d,
  count: sorted.filter(r => r.depth === d).length
}));
display(Plot.plot({
  width: Math.min(400, window.innerWidth - 80),
  height: 200,
  marginLeft: 80,
  x: {label: "Count"},
  y: {label: null},
  color: {domain: ["QUICK", "STANDARD", "DEEP"], range: ["#3b82f6", "#eab308", "#8b5cf6"]},
  marks: [
    Plot.barX(depthCounts, {x: "count", y: "depth", fill: "depth", tip: true}),
    Plot.ruleX([0])
  ]
}));
```
