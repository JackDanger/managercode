var	margin = {top: 30, right: 20, bottom: 30, left: 50},
	width = window.visualViewport.width - (margin.left + margin.right),
	height = window.visualViewport.height - (100 + margin.top + margin.bottom);

var	force_graph = d3.select("body")
	.append("svg")
		.attr("width", width)
		.attr("height", height)

var zoom = d3.zoom()
    .scaleExtent([1, 40])
    .translateExtent([[-100, -100], [width + 90, height + 100]])
    .on("zoom", zoomed);

var drag = d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);

function zoomed() {
  force_graph.attr("transform", d3.event.transform);
}
function dragstarted(d) {
  d3.event.sourceEvent.stopPropagation();
  d3.select(this).classed("dragging", true);
}
function dragged(d) {
  d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
}
function dragended(d) {
  d3.select(this).classed("dragging", false);
}

force_graph
    .call(zoom)
    .call(drag)
    .on("wheel", function() { d3.event.preventDefault(); })
    .on("zoom", zoomed);

drawForceGraph(force_graph)

function drawForceGraph(svg) {
  var color = d3.scaleOrdinal(d3.schemeCategory20);

  var force = d3.forceManyBody().strength(-50)
  var simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(function(d) { return d.id; }))
      .force("charge", force)
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("center", d3.forceCenter(width / 2, height / 2));

  d3.json("graphData.json", function(error, graph) {
    if (error) throw error;

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
        .attr("r", function(d) { return 4 + Math.sqrt(d.commits) })
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
        .attr("font-size", function(d) { return 14 + Math.log10(d.commits) })
        .text(function(d) { return d.id })
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))

    node.append("text")
        .attr("dx", 0)
        .attr("dy", ".65em")
        .text(function(d) { return d.id })
        .style("font-size", "20px")
        .style("fill", "#4393c3");

    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links)
        .distance(300)

    function ticked() {
      link
          .attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });

      node
          .attr("cx", function(d) { return d.x = Math.max(7, Math.min(width - 7, d.x)) })
          .attr("cy", function(d) { return d.y = Math.max(7, Math.min(height - 7, d.y)) });

      label
          .attr("x", function(d) { return d.x + 15 })
          .attr("y", function(d) { return d.y + 5});
    }
  });

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

