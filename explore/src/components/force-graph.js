import * as d3 from "d3";

/**
 * D3 force-directed graph component for the knowledge graph.
 *
 * @param {Array} entities - [{name, type, mentions}]
 * @param {Array} relationships - [{from, to, type, mentions}]
 * @param {Object} options - {width, height, onNodeClick}
 * @returns {SVGElement}
 */
export function ForceGraph(entities, relationships, {width = 800, height = 500, onNodeClick} = {}) {
  const types = [...new Set(entities.map(e => e.type))].sort();
  const color = d3.scaleOrdinal(d3.schemeTableau10).domain(types);

  // Build node/link data
  const nodeMap = new Map(entities.map(e => [e.name, e]));
  const nodes = entities.map(e => ({
    id: e.name,
    type: e.type,
    mentions: e.mentions ?? 1
  }));
  const links = relationships
    .filter(r => nodeMap.has(r.from) && nodeMap.has(r.to))
    .map(r => ({
      source: r.from,
      target: r.to,
      type: r.type,
      mentions: r.mentions ?? 1
    }));

  const relTypes = [...new Set(links.map(l => l.type))].sort();
  const linkColor = d3.scaleOrdinal(d3.schemeSet2).domain(relTypes);

  const svg = d3.create("svg")
    .attr("viewBox", [0, 0, width, height])
    .attr("width", width)
    .attr("height", height)
    .style("max-width", "100%")
    .style("height", "auto")
    .style("background", "#fafbfc")
    .style("border-radius", "12px")
    .style("border", "1px solid #e2e8f0");

  const g = svg.append("g");

  // Zoom
  const zoom = d3.zoom()
    .scaleExtent([0.3, 5])
    .on("zoom", (event) => g.attr("transform", event.transform));
  svg.call(zoom);

  // Simulation
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(80))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(d => nodeRadius(d) + 4));

  // Links
  const link = g.append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", d => linkColor(d.type))
    .attr("stroke-opacity", 0.5)
    .attr("stroke-width", d => Math.max(1, Math.sqrt(d.mentions)));

  // Nodes
  const node = g.append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", d => nodeRadius(d))
    .attr("fill", d => color(d.type))
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .style("cursor", "pointer")
    .call(drag(simulation));

  // Click handler
  node.on("click", (event, d) => {
    if (onNodeClick) onNodeClick(d.id);
  });

  // Tooltip
  node.append("title").text(d => `${d.id} (${d.type}, ${d.mentions} mentions)`);

  // Labels for high-mention nodes
  const labelThreshold = Math.max(2, d3.quantile(nodes.map(n => n.mentions).sort(d3.ascending), 0.7) ?? 2);
  const label = g.append("g")
    .selectAll("text")
    .data(nodes.filter(d => d.mentions >= labelThreshold))
    .join("text")
    .text(d => d.id.length > 20 ? d.id.slice(0, 18) + "..." : d.id)
    .attr("font-size", 10)
    .attr("font-family", "Inter, sans-serif")
    .attr("fill", "#334155")
    .attr("pointer-events", "none")
    .attr("dx", d => nodeRadius(d) + 3)
    .attr("dy", "0.35em");

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);
    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);
    label
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  });

  // Legend
  const legend = svg.append("g")
    .attr("transform", `translate(12, 12)`);

  types.forEach((t, i) => {
    const row = legend.append("g").attr("transform", `translate(0, ${i * 18})`);
    row.append("circle").attr("r", 5).attr("cx", 5).attr("cy", 0).attr("fill", color(t));
    row.append("text").attr("x", 14).attr("y", 4).text(t)
      .attr("font-size", 10).attr("font-family", "Inter, sans-serif").attr("fill", "#64748b");
  });

  return svg.node();
}

function nodeRadius(d) {
  return 4 + Math.sqrt(d.mentions ?? 1) * 3;
}

function drag(simulation) {
  return d3.drag()
    .on("start", (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (event, d) => {
      d.fx = event.x;
      d.fy = event.y;
    })
    .on("end", (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    });
}
