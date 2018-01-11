(function() {

var	contributor_force_graph_svg = d3.select("body")
	.append("svg")
		.attr("width", width)
		.attr("height", height)

var	repo_force_graph_svg = d3.select("body")
	.append("svg")
		.attr("width", width)
		.attr("height", height)

d3.json("contributorGraphData.json", function(error, graph) {
  if (error) throw error;
  drawForceGraph(contributor_force_graph_svg, graph)
})

d3.json("repoGraphData.json", function(error, graph) {
  if (error) throw error;
  drawForceGraph(repo_force_graph_svg, graph)
})

function drawForceGraph(svg, graph) {
  var force_graph = svg
      .append("g")
        .attr("width", width)
        .attr("height", height)

  force_graph.append("rect")
      .attr("fill", "none")
      .attr("pointer-events", "all")
      .attr("width", width)
      .attr("height", height)
      .call(d3.zoom()
          .scaleExtent([-100, 100])
          .on("zoom", function() { force_graph.attr("transform", d3.event.transform) }));

  var color = d3.scaleOrdinal(d3.schemeCategory20);

  var force = d3.forceManyBody().strength(-50)
  var simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(function(d) { return d.id; }))
      .force("charge", force)
      .force("center", d3.forceCenter(width / 2, height / 2));

  var link = svg.append("g")
      .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", function(d) { return Math.log10(d.value); });

  var node = svg.append("g")
    .attr("class", "node")
    .selectAll("circle")
    .data(graph.nodes)
    .enter()
      .append("circle")
      .attr("r", function(d) { return 4 + Math.sqrt(d.count) })
      .attr("fill", function(d) { return color(d.group); })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended))

  var label = svg.append("g")
    .attr("class", "labels")
    .selectAll("text")
    .data(graph.nodes)
    .enter()
      .append("text")
      .attr("class", "label")
      .attr("font-size", function(d) { return 14 + Math.log10(d.count) })
      .text(function(d) { return d.id.split('@')[0] })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended))

  node.append("text")
      .attr("dx", 0)
      .attr("dy", ".65em")
      .text(function(d) { return d.id.split('@')[0] })
      .style("font-size", "20px")
      .style("fill", "#4393c3");

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);

  simulation.force("link")
      .links(graph.links)
      .distance(200)

  function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x })
        .attr("cy", function(d) { return d.y });

    label
        .attr("x", function(d) { return d.x + 15 })
        .attr("y", function(d) { return d.y + 5});
  }

  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
}
})()
