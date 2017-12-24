/* Graphing team connectedness by Slack participation
 *
 * To generate:
 *   Run this file on the command line with a single argument that is a file* containing a Slack api token.
 *   The users in that Slack account will be graphed by connectedness.
 *
 * *avoiding env vars and command line arguments for security
 */
var fs = require('fs');
var Promise = require('bluebird');
const { WebClient } = require('@slack/client');

if (process.argv.length <= 2) {
    console.log("Usage: " + __filename + " path/to/file/containing/slack/apitoken");
    process.exit(1);
}
var secretfile = process.argv[2]
const slackToken = fs.readFileSync("./slack.token")
const client = new WebClient(slackToken, { retryConfig: { maxRequestConcurrency: 10 }})

const nintyDaysAgo = new Date() - 3600*24*90

const usernameMap = {}


client.users.list()
  .then(function(res) {
    res.members.forEach((user) => {
      if (!user.is_bot && user.name != 'slackbot') {
        usernameMap[user.id] = user.name
      }
    })
  })
  .then(function() {
    client.channels.list()
      .then(function(res) {
          return Promise.map(res.channels, function(channel) {
              return client.channels.history(channel.id)
                .then(function(r) {
                  r.messages.forEach(function(message) {
                    addMessage(channel.id, message.user)
                  })
                })
          })
      })
      .then(function() {
        fs.writeFile(__dirname + "/graphData.json", JSON.stringify(weightedGraph(), null, 2), function(err) {
          if(err) { return console.log(err) }
        }); 
        console.log("The contribution graph has been regenerated into ./slack-activity/graphData.js")
        console.log("Run a web server and open the html page to view results")
        console.log("python -m SimpleHTTPServer & (sleep 5 && open http://localhost:8000/git-contributors/contributors.html) && fg")
      })
    })
const users = {}
function addMessage(channel, username) {
  // Record a person contributing a commit to a repo
  if (!users[channel]) {
    users[channel] = {}
  }
  if (!users[channel][username]) {
    users[channel][username] = 1
  } else {
    users[channel][username]++
  }
}

function usernamePairNormalized(username1, username2) {
  // Use a consistent key for referencing pairs of users
  if (username1 < username2) {
    return [username1, username2]
  } else {
    return [username2, username1]
  }
}

function weightedGraph() {
  // Calculate the min message count of two users into the same channel.
  // If user1 wrote 12 messages and user2 wrote 15 then their overlap is 12
  var messageCount = {}
  var connectionCounts = {}

  for (var channel in users) {
    var messagesInChannel = {}
    for (var username in users[channel]) {
      if (!messagesInChannel[username]) {
        messagesInChannel[username] = 0
        messageCount[username] = 0
      }
      messageCount[username] += users[channel][username]
      messagesInChannel[username] += users[channel][username]
    }
    // Do a quadratic calculation to find the matrix overlap of all
    // contributors to this project
    for (var username1 in messagesInChannel) {
      for (var username2 in messagesInChannel) {
        if (username1 != username2) {
          var key = usernamePairNormalized(username1, username2)
          if (!connectionCounts[key]) {
            connectionCounts[key] = 0
          }
          // The connection between two people is the minimum of their
          // contributions to a single project, calculated for each project and
          // summed
          connectionCounts[key] += Math.min(messagesInChannel[username1], messagesInChannel[username2])
        }
      }
    }
  }
  var graph = {
    nodes: [],
    links: []
  }
  for (var username in messageCount) {
    var node = {
      id: usernameMap[username],
      group: 1,
      messages: messageCount[username],
    }
    graph.nodes.push(node)
  }
  for (var key in connectionCounts) {
    var userIdPair = key.split(',')
    var link = {
      source: usernameMap[userIdPair[0]],
      target: usernameMap[userIdPair[1]],
      value: connectionCounts[key]
    }
    graph.links.push(link)
  }
  return graph
}
