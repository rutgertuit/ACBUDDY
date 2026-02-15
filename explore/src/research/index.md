```js
display(html`<nav class="acb-nav">
  <a href="/" class="acb-nav-back">&larr; ACBUDDY</a>
  <span class="acb-nav-sep">|</span>
  <a href="/explore/">Archive</a>
  <a href="/explore/graph.html">Knowledge Graph</a>
  <a href="/explore/performance.html">Performance</a>
</nav>`);
```

# Research Deep Dive

```js
import {getArchive, getJobStatus, formatDuration, scoreColor, computeHumanMinutes, badgeClass} from "../components/api.js";

const params = new URLSearchParams(location.search);
const jobId = params.get("id");

let job = null;
let meta = null;

if (jobId) {
  job = await getJobStatus(jobId);
  const archive = await getArchive();
  meta = archive.find(d => d.job_id === jobId) ?? {};
}

const data = job ?? meta ?? {};
const score = data.synthesis_score ?? meta?.synthesis_score;
const timings = data.phase_timings ?? meta?.phase_timings ?? {};
const researchStats = data.research_stats ?? meta?.research_stats ?? {};
const studyProgress = data.study_progress ?? [];
const depth = data.depth ?? meta?.depth ?? "STANDARD";
const numStudies = studyProgress.length || meta?.num_studies || 0;
const effort = computeHumanMinutes(researchStats, numStudies, depth);
```

```js
if (!jobId) {
  display(html`<div class="empty-state">
    <p>No research ID specified. Go to the <a href="/explore/">Archive</a> and click a research run.</p>
  </div>`);
}
```

```js
if (jobId && data.query) {
  display(html`<div style="margin-bottom:1.5rem;">
    <span class="${badgeClass(depth)}">${depth}</span>
    <h2 style="margin:0.5rem 0 0.25rem;">${data.query}</h2>
    <span style="color:var(--acb-text-muted);font-size:0.85rem;">Job ${jobId} &middot; ${data.status ?? "completed"}</span>
  </div>`);
}
```

## Quality Scorecard

```js
if (jobId && score != null) {
  const sc = scoreColor(score);
  const dimScores = data.synthesis_scores ?? meta?.synthesis_scores ?? {};
  const dims = Object.entries(dimScores).filter(([, v]) => typeof v === "number");

  display(html`<div class="card-row">
    <div class="card" style="text-align:center;max-width:200px;">
      <div class="label">Overall Score</div>
      <div class="big ${sc}">${score.toFixed(1)}<span class="unit">/ 10</span></div>
      ${data.refinement_rounds != null || meta?.refinement_rounds != null
        ? html`<div style="color:var(--acb-text-muted);font-size:0.8rem;">${data.refinement_rounds ?? meta?.refinement_rounds ?? 0} refinement rounds</div>`
        : ""}
    </div>
  </div>`);

  if (dims.length > 0) {
    display(Plot.plot({
      width: Math.min(600, window.innerWidth - 80),
      height: Math.max(120, dims.length * 28),
      marginLeft: 150,
      x: {label: "Score", domain: [0, 10]},
      y: {label: null},
      marks: [
        Plot.barX(dims, {
          x: d => d[1],
          y: d => d[0],
          fill: d => d[1] >= 7 ? "#22c55e" : d[1] >= 5 ? "#eab308" : "#ef4444",
          tip: true
        }),
        Plot.ruleX([7], {stroke: "#22c55e", strokeDasharray: "4,4"}),
        Plot.ruleX([0])
      ]
    }));
  }
}
```

## Phase Timeline

```js
if (jobId && Object.keys(timings).length > 0) {
  const phaseOrder = ["query_analysis", "study_planning", "study_research", "synthesis", "verification", "qa_anticipation", "master_synthesis", "strategic_analysis"];
  const phases = phaseOrder
    .filter(p => timings[p])
    .map(p => ({
      phase: p.replace(/_/g, " "),
      duration: timings[p].duration ?? 0,
      start: timings[p].start ?? 0
    }))
    .filter(p => p.duration > 0);

  for (const [k, v] of Object.entries(timings)) {
    if (!phaseOrder.includes(k) && v.duration > 0) {
      phases.push({phase: k.replace(/_/g, " "), duration: v.duration, start: v.start ?? 0});
    }
  }

  if (phases.length > 0) {
    const minStart = Math.min(...phases.map(p => p.start));
    const ganttData = phases.map(p => ({
      phase: p.phase,
      start: p.start - minStart,
      end: (p.start - minStart) + p.duration,
      duration: p.duration
    }));

    display(Plot.plot({
      width: Math.min(800, window.innerWidth - 80),
      height: Math.max(120, ganttData.length * 30),
      marginLeft: 140,
      x: {label: "Seconds from start"},
      y: {label: null},
      marks: [
        Plot.barX(ganttData, {
          x1: "start",
          x2: "end",
          y: "phase",
          fill: "#0dccf2",
          tip: true
        }),
        Plot.text(ganttData, {
          x: "end",
          y: "phase",
          text: d => formatDuration(d.duration),
          dx: 6,
          textAnchor: "start",
          fontSize: 11,
          fill: "#64748b"
        }),
        Plot.ruleX([0])
      ]
    }));
  }
}
```

## Research Effort

```js
if (jobId) {
  display(html`<div class="card-row">
    <div class="card">
      <div class="label">Web Searches</div>
      <div class="big">${researchStats.web_searches ?? 0}</div>
    </div>
    <div class="card">
      <div class="label">Pages Read</div>
      <div class="big">${researchStats.pages_read ?? 0}</div>
    </div>
    <div class="card">
      <div class="label">Reasoning Calls</div>
      <div class="big">${researchStats.reasoning_calls ?? 0}</div>
    </div>
    <div class="card">
      <div class="label">Human-Hour Equivalent</div>
      <div class="big">${effort.hours}<span class="unit">hrs</span></div>
    </div>
  </div>`);
}
```

## Study Progress

```js
if (jobId && studyProgress.length > 0) {
  const rows = studyProgress.map(s => `<tr>
    <td>${s.title ?? "—"}</td>
    <td>${s.status ?? "—"}</td>
    <td>${s.rounds ?? "—"}</td>
  </tr>`).join("");
  display(html`<table class="data-table">
    <thead><tr><th>Study</th><th>Status</th><th>Rounds</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`);
}
```

```js
if (jobId && data.result_url) {
  display(html`<div style="margin-top:1.5rem;">
    <a href="${data.result_url}" target="_blank" style="color:var(--acb-primary);font-weight:600;">View Full HTML Report &rarr;</a>
  </div>`);
}
```
