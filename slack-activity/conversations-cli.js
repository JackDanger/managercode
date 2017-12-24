/* Graphing team connectedness by Slack participation
 *
 * To generate:
 *   Run this file on the command line with a single argument that is a file* containing a Slack api token.
 *   The users in that Slack account will be graphed by connectedness.
 *
 * *avoiding env vars and command line arguments for security
 */
var fs = require('fs');
const { WebClient } = require('@slack/client');
 
if (process.argv.length <= 2) {
    console.log("Usage: " + __filename + " path/to/file/containing/slack/apitoken");
    process.exit(1);
}
var secretfile = process.argv[2]
var slackToken = fs.readFileSync(secretfile)

var users = {}
function addConversation(channel, user) {
  // Record a person contributing a commit to a repo
  if (!users[channel]) {
    users[channel] = {}
  }
  if (!users[channel][user]) {
    users[channel][user] = 1
  } else {
    users[channel][user]++
  }
}

function usernamePairNormalized(username1, username2) {
  // Use a consistent key for referencing pairs of emails
  if (username1 == "" || username2 == "") {
    console.log(username1, username2)
  }
  if (username1 < username2) {
    return [username1, username2]
  } else {
    return [username2, username1]
  }
}

emailDomains = []
function groupFromEmailDomain(email) {
  // Given an email address, return an integer to use as a clustering group
  // decided by retrieving the index of the domain in the list of all domains
  // that have yet been calculated by this function
  var parts = email.split('@')
  if (parts.length < 2) {
    return 1  // incalculable case
  }
  var domain = parts[1]
  if (emailDomains.indexOf(domain) < 0) {
    emailDomains.push(domain)
  }
  return emailDomains.indexOf(domain) + 2 // always greater than the incalculable case
}

function weightedGraph() {
  // Calculate the min message count of two users into the same channel.
  // If user1 wrote 12 messages and user2 wrote 15 then their overlap is 12
  var messageCount = {}
  var connectionCounts = {}

  for (var project in contributors) {
    var messagesInChannel = {}
    for (var email in contributors[project]) {
      if (!messagesInChannel[email]) {
        messagesInChannel[email] = 0
        messageCount[email] = 0
      }
      messageCount[email] += contributors[project][email]
      messagesInChannel[email] += contributors[project][email]
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
  for (var email in messageCount) {
    var node = {
      id: email,
      group: groupFromEmailDomain(email),
      commits: messageCount[email],
    }
    graph.nodes.push(node)
  }
  for (var emailPairKey in connectionCounts) {
    var emailPair = emailPairKey.split(',')
    var link = {
      source: emailPair[0],
      target: emailPair[1],
      value: connectionCounts[emailPair]
    }
    graph.links.push(link)
  }
  return graph
}

glob(repo_list + "/*/.git", function(err, files) {
  for (var idx in files) {
    var basedir = files[idx].replace('/.git', '')
    console.log("repo: " + basedir)
    var buff = execSync("git log --format='%ae' --since='3 months ago'", {cwd: basedir})
    stdout = buff.toString('utf-8')
    if (stdout.length) {
      var lines = stdout.split('\n')
      for (var idx in lines) {
        if (lines[idx].length > 2) {
          // record that a non-blank email has made a git commit
          addContribution(basedir, lines[idx])
        }
      }
    }
  }
  fs.writeFile("./graphData.json", JSON.stringify(weightedGraph(), null, 2), function(err) {
    if(err) {
        return console.log(err);
    }
  }); 
  console.log("The contribution graph has been regenerated into ./graphData.js")
  console.log("Run a web server and open the html page to view results")
  console.log("python -m SimpleHTTPServer & (sleep 5 && open http://localhost:8000/git-contributors/contributors.html) && fg")
})
