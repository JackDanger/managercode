// This script is designed to run in the browser console on Slack's web interface
// It assumes you're already authenticated and in a Slack workspace

// Helper function to fetch messages from a channel
async function fetchChannelHistory(channelId, oldest) {
  return new Promise((resolve) => {
    let messages = [];
    let fetchMore = async (cursor = null) => {
      let url = `/api/conversations.history?channel=${channelId}&limit=1000&inclusive=true&oldest=${oldest}`;
      if (cursor) url += `&cursor=${cursor}`;
      
      let response = await fetch(url, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-Slack-Version': '4',  // This might need updating based on Slack's current version
        },
        method: 'POST',
      });
      let data = await response.json();
      
      messages = messages.concat(data.messages);
      
      if (data.has_more && data.response_metadata.next_cursor) {
        await fetchMore(data.response_metadata.next_cursor);
      } else {
        resolve(messages);
      }
    };
    fetchMore();
  });
}

// Function to build the graph from messages
function buildGraph(messages) {
  const graph = { nodes: new Set(), links: {} };
  const userNameCache = {};

  messages.forEach(message => {
    if (!message.user) return;  // Skip messages without a user (e.g., bot messages)

    const sender = message.user;
    graph.nodes.add(sender);
    
    // Extract mentions from message text
    const mentions = (message.text.match(/<@([A-Z0-9]+)>/g) || [])
      .map(mention => mention.slice(2, -1));
    
    mentions.forEach(mention => {
      graph.nodes.add(mention);
      const key = `${sender}-${mention}`;
      graph.links[key] = (graph.links[key] || 0) + 1;
    });
  });

  // Convert graph to D3 format
  return {
    nodes: Array.from(graph.nodes).map(id => ({ id })),
    links: Object.entries(graph.links).map(([key, value]) => {
      const [source, target] = key.split('-');
      return { source, target, value };
    })
  };
}

// Function to create D3 visualization
function createVisualization(graph) {
  // Load D3 library
  const script = document.createElement('script');
  script.src = 'https://d3js.org/d3.v7.min.js';
  document.head.appendChild(script);

  script.onload = () => {
    const width = 800;
    const height = 600;

    // Create SVG element
    const svg = d3.select("body").append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("position", "fixed")
      .style("top", "0")
      .style("left", "0")
      .style("z-index", "9999");

    const simulation = d3.forceSimulation(graph.nodes)
      .force("link", d3.forceLink(graph.links).id(d => d.id))
      .force("charge", d3.forceManyBody())
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
      .selectAll("line")
      .data(graph.links)
      .enter().append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", d => Math.sqrt(d.value));

    const node = svg.append("g")
      .selectAll("circle")
      .data(graph.nodes)
      .enter().append("circle")
      .attr("r", 5)
      .attr("fill", "#69b3a2")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    node.append("title")
      .text(d => d.id);

    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
    });

    function dragstarted(event) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  };
}

// Main function to run the entire process
async function generateSlackNetworkGraph() {
  // Get all channel IDs from the sidebar
  const channelElements = document.querySelectorAll('[data-qa="channel_sidebar_name_button"]');
  const channelIds = Array.from(channelElements).map(el => el.getAttribute('data-qa-channel-sidebar-channel-id'));

  let allMessages = [];
  const thirtyDaysAgo = Math.floor(Date.now() / 1000) - (30 * 24 * 60 * 60);

  for (let channelId of channelIds) {
    const messages = await fetchChannelHistory(channelId, thirtyDaysAgo);
    allMessages = allMessages.concat(messages);
  }

  const graph = buildGraph(allMessages);
  createVisualization(graph);
}

// Run the main function
generateSlackNetworkGraph();
