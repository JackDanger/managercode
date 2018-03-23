#!/usr/bin/env node
/* Graphing team connectedness by Slack participation
 *
 * To generate:
 *   Run this file on the command line with a single argument that is a file* containing a Slack api token.
 *   The users in that Slack account will be graphed by connectedness.
 *
 * *avoiding env vars and command line arguments for security
 */
const fs = require('fs');
const os = require('os');
const Promise = require('bluebird');
const { WebClient } = require('@slack/client');
var slackToken;

if (process.env.SLACK_TOKEN) {
  slackToken = process.env.SLACK_TOKEN
} else {
  if (process.argv.length <= 2) {
      console.log("Usage: " + __filename + " path/to/file/containing/slack/apitoken");
      process.exit(1);
  }
  var secretfile = process.argv[2]
  slackToken = fs.readFileSync(secretfile)
}
const client = new WebClient(slackToken)

const nintyDaysAgo = new Date() - 3600*24*90

var onlyTheseEmails;
if (process.env.ONLY_THESE_EMAILS) {
  onlyTheseEmails = JSON.parse(process.env.ONLY_THESE_EMAILS)
  console.log("scoping to only", onlyTheseEmails.length, "email addresses")
}

const userMap = {}


client.users.list()
  .then(function(res) {
    res.members.forEach((user) => { userMap[user.id] = user })
  })
  .then(function() {
    client.channels.list()
      .then((res) => {
        return res.channels
      })
      .mapSeries((channel) => {
        return client.channels.history(channel.id)
          .then(function(r) {
            console.log("start")
            var i = 0;
            while (i < 1000000000) {
              i++
            }
            console.log("iterated")
            r.messages.forEach(function(message) {
              if (message.user && message.user != 'USLACKBOT') {
                addMessage(channel.id, message.user)
              }
            })
          })
      })
      .then(function() {
        fs.writeFile(__dirname + "/graphData.json", JSON.stringify(weightedGraph(), null, 2), function(err) {
          if(err) { return console.log(err) }
        }); 
        console.log("The contribution graph has been regenerated into ./" + __dirname + "/graphData.js")
        console.log("Run a web server and open the html page to view results")
        console.log("python -m SimpleHTTPServer & (sleep 5 && open http://localhost:8000/" + __dirname + "/index.html) && fg")
      })
    })

const users = {}
function addMessage(channel, username) {
  // Check if we're limiting to only a subset of users
  if (onlyTheseEmails) {
    for (var email in onlyTheseEmails) {
      if (userMap[username].email == email) {
        return
      }
    }
  }
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

function stringPairNormalized(str1, str2) {
  // Use a consistent key for referencing pairs of strings
  if (str1 < str2) {
    return [str1, str2]
  } else {
    return [str2, str1]
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
          var key = stringPairNormalized(username1, username2)
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
      id: userMap[username].profile.display_name || userMap[username].name,
      group: 1,
      count: messageCount[username],
    }
    graph.nodes.push(node)
  }
  for (var key in connectionCounts) {
    var userIdPair = key.split(',')
    var user1 = userMap[userIdPair[0]]
    var user2 = userMap[userIdPair[1]]
    if (user1.name != 'slackbot' && user2.name != 'slackbot' && !user1.is_bot && !user2.is_bot) {
      var link = {
        source: user1.profile.display_name || user1.name,
        target: user2.profile.display_name || user2.name,
        value: connectionCounts[key]
      }
      graph.links.push(link)
    }
  }
  return graph
}
