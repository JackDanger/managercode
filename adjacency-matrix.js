(function() {

var	adjacency_svg = d3.select("body")
	.append("svg")
		.attr("width", width)
		.attr("height", height)

d3.json('graphData.json', function(data) {
  data.nodes.sort(function(a, b) { return d3.descending(a.count, b.count) } )
  console.log("data", data.nodes)
  const adjacencyMatrix = d3AdjacencyMatrixLayout();
  console.log('adjacencyMatrix', adjacencyMatrix);
  console.log('d3', d3);

  adjacencyMatrix
    .size([870,870])
    .nodes(data.nodes)
    .links(data.links)
    .directed(false)
    .nodeID(function(d) { return d.id });

  const matrixData = adjacencyMatrix();

  const someColors = d3.scaleOrdinal()
    .range(d3.schemeCategory20b);

  adjacency_svg
    .append('g')
      .attr('transform', 'translate(80,80)')
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
        .style('fill-opacity', d => 1 - (1 / Math.log(d.source.count + d.target.count)));

  d3.select('#adjacencyG')
    .call(adjacencyMatrix.xAxis);

  d3.select('#adjacencyG')
    .call(adjacencyMatrix.yAxis);
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
        weight: edgeWeight(edge)
      };
      if (typeof edge.source === 'number') {
        constructedEdge.source = nodes[edge.source];
      }
      if (typeof edge.target === 'number') {
        constructedEdge.target = nodes[edge.target];
      }
      var id = nodeID(constructedEdge.source) + '-' + nodeID(constructedEdge.target);

      if (directed === false && constructedEdge.source.sortedIndex < constructedEdge.target.sortedIndex) {
        id = nodeID(constructedEdge.target) + '-' + nodeID(constructedEdge.source);
      }
      if (!edgeHash[id]) {
        edgeHash[id] = constructedEdge;
      } else {
        edgeHash[id].weight = edgeHash[id].weight + constructedEdge.weight;
      }
    });

    nodes.forEach(function (sourceNode, a) {
      nodes.forEach(function (targetNode, b) {
        var grid = {
          id: nodeID(sourceNode) + '-' + nodeID(targetNode),
          source: sourceNode,
          target: targetNode,
          x: xScale(b),
          y: yScale(a),
          weight: 0,
          height: nodeHeight,
          width: nodeWidth
        };
        var edgeWeight = 0;
        if (edgeHash[grid.id]) {
          edgeWeight = edgeHash[grid.id].weight;
          grid.weight = edgeWeight;
        }
        if (directed === true || b < a) {
          matrix.push(grid);
          if (directed === false) {
            var mirrorGrid = {
              id: nodeID(sourceNode) + '-' + nodeID(targetNode),
              source: sourceNode,
              target: targetNode,
              x: xScale(a),
              y: yScale(b),
              weight: 0,
              height: nodeHeight,
              width: nodeWidth
            };
            mirrorGrid.weight = edgeWeight;
            matrix.push(mirrorGrid);
          }
        }
      });
    });

    console.log('matrix', matrix, matrix.length);

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
    var nameScale = d3.scalePoint().domain(nodes.map(nodeID)).range([0, size[0]]).padding(1);

    var xAxis = d3.axisTop().scale(nameScale).tickSize(4);

    calledG.append('g').attr('class', 'am-xAxis am-axis').call(xAxis).selectAll('text').style('text-anchor', 'end').attr('transform', 'translate(-10,-10) rotate(90)');
  };

  matrix.yAxis = function (calledG) {
    var nameScale = d3.scalePoint().domain(nodes.map(nodeID)).range([0, size[1]]).padding(1);

    var yAxis = d3.axisLeft().scale(nameScale).tickSize(4);

    calledG.append('g').attr('class', 'am-yAxis am-axis').call(yAxis);
  };

  return matrix;
} 

})()
