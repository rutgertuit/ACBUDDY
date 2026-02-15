/**
 * Shared API fetch helpers with 60-second client-side cache.
 */

const _cache = new Map();
const CACHE_TTL = 60_000; // 60 seconds

async function cachedFetch(url) {
  const now = Date.now();
  const entry = _cache.get(url);
  if (entry && now - entry.ts < CACHE_TTL) return entry.data;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  _cache.set(url, {ts: now, data});
  return data;
}

// ── API fetchers ──

export async function getArchive() {
  const d = await cachedFetch("/api/archive");
  return d.results ?? [];
}

export async function getGraph() {
  return cachedFetch("/api/graph");
}

export async function getMemory() {
  return cachedFetch("/api/memory");
}

export async function getStats() {
  return cachedFetch("/api/stats");
}

export async function getTimingEstimates() {
  return cachedFetch("/api/timing-estimates");
}

export async function getJobStatus(jobId) {
  // Don't cache individual status calls
  const resp = await fetch(`/api/status/${jobId}`);
  if (!resp.ok) return null;
  return resp.json();
}

export async function getEntityConnections(name) {
  const resp = await fetch(`/api/graph/entity/${encodeURIComponent(name)}`);
  if (!resp.ok) return null;
  return resp.json();
}

// ── Computation helpers ──

/**
 * Mirror of research_stats.py:compute_human_hours
 */
export function computeHumanMinutes(stats, numStudies = 0, depth = "STANDARD") {
  if (!stats) return {searching: 0, reading: 0, analyzing: 0, writing: 0, total: 0, hours: 0};
  const searching = ((stats.web_searches ?? 0) + (stats.news_searches ?? 0) + (stats.grok_queries ?? 0)) * 8;
  const reading = (stats.pages_read ?? 0) * 5 + (stats.news_articles ?? 0) * 3;
  const analyzing = (stats.reasoning_calls ?? 0) * 15 + numStudies * 45;
  const writing = depth.toUpperCase() === "DEEP" ? 120 : 30;
  const total = searching + reading + analyzing + writing;
  return {searching, reading, analyzing, writing, total, hours: Math.round(total / 60 * 10) / 10};
}

export function formatDuration(seconds) {
  if (seconds == null || seconds <= 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm > 0 ? `${h}h ${rm}m` : `${h}h`;
}

export function scoreColor(score) {
  if (score == null) return "";
  if (score >= 7) return "score-high";
  if (score >= 5) return "score-mid";
  return "score-low";
}

export function badgeClass(depth) {
  const d = (depth ?? "").toLowerCase();
  if (d === "quick") return "badge badge-quick";
  if (d === "standard") return "badge badge-standard";
  if (d === "deep") return "badge badge-deep";
  return "badge";
}
