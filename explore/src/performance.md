```js
display(html`<nav class="acb-nav">
  <a href="/" class="acb-nav-back">&larr; Luminary</a>
  <span class="acb-nav-sep">|</span>
  <a href="/explore/">Archive</a>
  <a href="/explore/graph.html">Knowledge Graph</a>
  <a href="/explore/performance.html" class="acb-nav-active">Performance</a>
</nav>`);
```

# Performance Analytics

```js
import {getArchive, getTimingEstimates, getMemory, formatDuration, scoreColor, computeHumanMinutes} from "./components/api.js";

const archive = await getArchive();
const timingEst = await getTimingEstimates();
const memory = await getMemory();

// Only DEEP runs with timings
const deepRuns = archive
  .filter(d => d.depth === "DEEP" && d.phase_timings && Object.keys(d.phase_timings).length > 0)
  .map(d => ({...d, date: new Date(d.completed_at || d.created_at)}))
  .sort((a, b) => a.date - b.date);

// All runs with dates
const allRuns = archive
  .map(d => ({...d, date: new Date(d.completed_at || d.created_at)}))
  .sort((a, b) => a.date - b.date);
```

```js
display(html`<div class="card-row">
  <div class="card">
    <div class="label">DEEP Runs Analyzed</div>
    <div class="big">${deepRuns.length}</div>
  </div>
  <div class="card">
    <div class="label">Avg Total Duration</div>
    <div class="big">${formatDuration(timingEst.total_average ?? 0)}</div>
  </div>
  <div class="card">
    <div class="label">Memory Entries</div>
    <div class="big">${memory.stats?.total_entries ?? memory.entries?.length ?? 0}</div>
  </div>
</div>`);
```

## Phase Duration Box Plots

```js
if (deepRuns.length > 0) {
  const phaseData = [];
  for (const run of deepRuns) {
    for (const [phase, timing] of Object.entries(run.phase_timings)) {
      if (timing.duration > 0) {
        phaseData.push({phase: phase.replace(/_/g, " "), duration: timing.duration});
      }
    }
  }

  if (phaseData.length > 0) {
    display(Plot.plot({
      width: Math.min(900, window.innerWidth - 80),
      height: 300,
      marginLeft: 140,
      x: {label: "Duration (seconds)"},
      y: {label: null},
      marks: [
        Plot.boxX(phaseData, {x: "duration", y: "phase", fill: "#0dccf2", fillOpacity: 0.6}),
        Plot.ruleX([0])
      ]
    }));
  }
} else {
  display(html`<div class="empty-state">No DEEP runs with timing data available yet.</div>`);
}
```

## Total Duration Trend

```js
if (deepRuns.length > 0) {
  const durationTrend = deepRuns.map(d => {
    const total = Object.values(d.phase_timings).reduce((a, t) => a + (t.duration || 0), 0);
    return {date: d.date, duration: total, query: d.query?.slice(0, 40)};
  });

  display(Plot.plot({
    width: Math.min(900, window.innerWidth - 80),
    height: 250,
    marginLeft: 60,
    x: {label: null},
    y: {label: "Duration (seconds)"},
    marks: [
      Plot.line(durationTrend, {x: "date", y: "duration", stroke: "#0dccf2", strokeWidth: 2}),
      Plot.dot(durationTrend, {x: "date", y: "duration", fill: "#0dccf2", tip: true}),
      Plot.ruleY([0])
    ]
  }));
}
```

## Quality Score Trend

```js
const scoredRuns = allRuns.filter(d => d.synthesis_score != null && d.synthesis_score > 0);
if (scoredRuns.length > 0) {
  display(Plot.plot({
    width: Math.min(900, window.innerWidth - 80),
    height: 250,
    marginLeft: 40,
    x: {label: null},
    y: {label: "Quality Score", domain: [0, 10]},
    marks: [
      Plot.ruleY([7], {stroke: "#22c55e", strokeDasharray: "4,4", strokeWidth: 1.5}),
      Plot.line(scoredRuns, {x: "date", y: "synthesis_score", stroke: "#8b5cf6", strokeWidth: 2}),
      Plot.dot(scoredRuns, {
        x: "date",
        y: "synthesis_score",
        fill: d => d.synthesis_score >= 7 ? "#22c55e" : d.synthesis_score >= 5 ? "#eab308" : "#ef4444",
        r: 5,
        tip: true
      }),
      Plot.ruleY([0])
    ]
  }));
} else {
  display(html`<div class="empty-state">No quality scores available yet.</div>`);
}
```

## Research Effort (Stacked Bar)

```js
if (deepRuns.length > 0) {
  const effortData = [];
  for (const run of deepRuns) {
    const stats = run.research_stats ?? {};
    const label = run.query?.slice(0, 30) ?? run.job_id;
    effortData.push({run: label, metric: "Web Searches", count: stats.web_searches ?? 0});
    effortData.push({run: label, metric: "Pages Read", count: stats.pages_read ?? 0});
    effortData.push({run: label, metric: "Reasoning Calls", count: stats.reasoning_calls ?? 0});
  }

  if (effortData.some(d => d.count > 0)) {
    display(Plot.plot({
      width: Math.min(900, window.innerWidth - 80),
      height: Math.max(200, deepRuns.length * 35),
      marginLeft: 200,
      x: {label: "Count"},
      y: {label: null},
      color: {domain: ["Web Searches", "Pages Read", "Reasoning Calls"], range: ["#3b82f6", "#22c55e", "#f59e0b"]},
      marks: [
        Plot.barX(effortData, Plot.stackX({x: "count", y: "run", fill: "metric", tip: true})),
        Plot.ruleX([0])
      ]
    }));
  }
}
```

## Memory Growth

```js
const entries = memory.entries ?? [];
if (entries.length > 0) {
  const memDates = entries
    .filter(e => e.created_at)
    .map(e => ({date: new Date(e.created_at), type: e.type}))
    .sort((a, b) => a.date - b.date);

  const cumulative = memDates.map((d, i) => ({date: d.date, count: i + 1}));

  if (cumulative.length > 0) {
    display(Plot.plot({
      width: Math.min(900, window.innerWidth - 80),
      height: 200,
      marginLeft: 40,
      x: {label: null},
      y: {label: "Total Entries"},
      marks: [
        Plot.areaY(cumulative, {x: "date", y: "count", fill: "#0dccf2", fillOpacity: 0.2}),
        Plot.line(cumulative, {x: "date", y: "count", stroke: "#0dccf2", strokeWidth: 2}),
        Plot.ruleY([0])
      ]
    }));
  }
} else {
  display(html`<div class="empty-state">No memory entries yet.</div>`);
}
```

## Human-Hour Equivalents

```js
if (deepRuns.length > 0) {
  const humanData = deepRuns.map(d => {
    const stats = d.research_stats ?? {};
    const ns = d.num_studies ?? 0;
    const eff = computeHumanMinutes(stats, ns, "DEEP");
    return {
      run: d.query?.slice(0, 30) ?? d.job_id,
      hours: eff.hours
    };
  });

  display(Plot.plot({
    width: Math.min(900, window.innerWidth - 80),
    height: Math.max(200, humanData.length * 35),
    marginLeft: 200,
    x: {label: "Human-Hour Equivalent"},
    y: {label: null},
    marks: [
      Plot.barX(humanData, {x: "hours", y: "run", fill: "#0dccf2", tip: true}),
      Plot.ruleX([0])
    ]
  }));
}
```
