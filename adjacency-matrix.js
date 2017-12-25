(function() {

var	adjacency_svg = d3.select("body")
	.append("svg")
		.attr("width", width)
		.attr("height", height)
var adjacency = adjacency_svg
	.append("g")
		.attr("width", width)
		.attr("height", height)

d3.json('graphData.json', function(data) {
  data.nodes.sort(function(a, b) { return b.weight < a.weight } )
  const adjacencyMatrix = d3AdjacencyMatrixLayout();

  adjacencyMatrix
    .size([width - 200, height - 100])
    .nodes(data.nodes)
    .links(data.links)
    .directed(false)
    .nodeID(function(d) { return d.id });

  const matrixData = adjacencyMatrix();

  const someColors = d3.scaleOrdinal()
    .range(d3.schemeCategory20b);

  adjacency
    .append("g")
    .attr('transform', 'translate(120,120)')
    .attr('id', 'adjacencyG')
    .selectAll('rect')
    .data(matrixData)
    .enter()
    .append('rect')
      .attr('width', d => d.width)
      .attr('height', d => d.height)
      .attr('x', d => d.x)
      .attr('y', d => d.y)
      .style('stroke', 'black')
      .style('stroke-width', '1px')
      .style('stroke-opacity', .1)
      .style('fill', d => someColors(d.source.group))
      .attr('weight', d => d.weight)
      .style('fill-opacity', d => 1 - (1 / (d.weight || 1)));

  d3.select('#adjacencyG')
    .call(adjacencyMatrix.xAxis);

  d3.select('#adjacencyG')
    .call(adjacencyMatrix.yAxis);

  adjacency.append("rect")
      .attr("fill", "none")
      .attr("pointer-events", "all")
      .attr("width", width)
      .attr("height", height)
      .call(d3.zoom()
          .scaleExtent([-100, 100])
          .on("zoom", function() { adjacency.attr("transform", d3.event.transform) }));

})

function d3AdjacencyMatrixLayout () {
  var directed = true;
  var size = [1, 1];
  var nodes = [];
  var edges = [];
  var edgeWeight = function edgeWeight(d) {
    return 1;
  };
  var nodeID = function nodeID(d) {
    return d.id;
  };

  function matrix() {
    var width = size[0];
    var height = size[1];
    var n = nodes.length
    var nodeWidth = width / n;
    var nodeHeight = height / n;
    // const constructedMatrix = [];
    var matrix = [];
    var edgeHash = {};
    var xScale = d3.scaleLinear().domain([0, nodes.length]).range([0, width]);
    var yScale = d3.scaleLinear().domain([0, nodes.length]).range([0, height]);

    nodes.forEach(function (node, i) {
      node.sortedIndex = i;
    });

    edges.forEach(function (edge) {
      var constructedEdge = {
        source: edge.source,
        target: edge.target,
        weight: edge.value
      };
      if (typeof edge.source === 'number') {
        constructedEdge.source = nodes[edge.source];
      }
      if (typeof edge.target === 'number') {
        constructedEdge.target = nodes[edge.target];
      }

      var id = keyPair(constructedEdge.source, constructedEdge.target);
      constructedEdge.id = id

      edgeHash[id] = constructedEdge;
    });

    nodes.forEach(function (sourceNode, a) {
      nodes.forEach(function (targetNode, b) {
        var grid = {
          id: keyPair(nodeID(sourceNode), nodeID(targetNode)),
          source: sourceNode,
          target: targetNode,
          x: xScale(b),
          y: yScale(a),
          height: nodeHeight,
          width: nodeWidth
        };
        if (edgeHash[grid.id]) {
          grid.weight = edgeHash[grid.id].weight;
        } else {
          grid.weight = 0;
        }
        if (directed === true || b < a) {
          matrix.push(grid);
          if (directed === false) {
            var mirrorGrid = {
              id: keyPair(nodeID(sourceNode), nodeID(targetNode)),
              source: sourceNode,
              target: targetNode,
              x: xScale(a),
              y: yScale(b),
              weight: grid.weight,
              height: nodeHeight,
              width: nodeWidth
            };
            matrix.push(mirrorGrid);
          }
        }
      });
    });

    return matrix;
  }

  matrix.directed = function (x) {
    if (!arguments.length) return directed;
    directed = x;
    return matrix;
  };

  matrix.size = function (x) {
    if (!arguments.length) return size;
    size = x;
    return matrix;
  };

  matrix.nodes = function (x) {
    if (!arguments.length) return nodes;
    nodes = x;
    return matrix;
  };

  matrix.links = function (x) {
    if (!arguments.length) return edges;
    edges = x;
    return matrix;
  };

  matrix.edgeWeight = function (x) {
    if (!arguments.length) return edgeWeight;
    if (typeof x === 'function') {
      edgeWeight = x;
    } else {
      edgeWeight = function edgeWeight() {
        return x;
      };
    }
    return matrix;
  };

  matrix.nodeID = function (x) {
    if (!arguments.length) return nodeID;
    if (typeof x === 'function') {
      nodeID = x;
    }
    return matrix;
  };

  matrix.xAxis = function (calledG) {
    var nameScale = d3.scalePoint().domain(nodes.map(nodeID)).range([1, size[0]]).padding(5);

    var xAxis = d3.axisTop().scale(nameScale).tickSize(4);

    calledG.append('g').attr('class', 'am-xAxis am-axis').call(xAxis).selectAll('text').style('text-anchor', 'end').attr('transform', 'translate(-10,-10) rotate(90)');
  };

  matrix.yAxis = function (calledG) {
    var nameScale = d3.scalePoint().domain(nodes.map(nodeID)).range([2, size[1] + 2]).padding(5);

    var yAxis = d3.axisLeft().scale(nameScale).tickSize(4);

    calledG.append('g').attr('class', 'am-yAxis am-axis').call(yAxis);
  };

  return matrix;
} 

function keyPair(str1, str2) {
  // Use a consistent key for referencing pairs of strings
  if (str1 < str2) {
    return str1 + '-' + str2
  } else {
    return str2 + '-' + str1
  }
}

})()
