/* Graphing team collaboration by git contributions
 *
 * To generate:
 *   Run this file on the command line with a target directory which contains
 *   any number of git repositories. This script will calculate author
 *   contributions and output them as two lists of edges and vertices weighted
 *   by the square root of connections between committers
 */
const { execSync } = require('child_process'); var glob = require('glob');
var fs = require('fs');
 
if (process.argv.length <= 2) {
    console.log("Usage: " + __filename + " path/to/directory");
    process.exit(1);
}
// cd /first/argument
var repo_list = process.argv[2]


var contributors = {}
function addContribution(project, email) {
  // Record a person contributing a commit to a repo
  if (!contributors[project]) {
    contributors[project] = {}
  }
  if (!contributors[project][email]) {
    contributors[project][email] = 1
  } else {
    contributors[project][email]++
  }
}

function emailPairNormalized(email1, email2) {
  // Use a consistent key for referencing pairs of emails
  if (email1 == "" || email2 == "") {
    console.log(email1, email2)
  }
  if (email1 < email2) {
    return [email1, email2]
  } else {
    return [email2, email1]
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
  // Calculate the connections between committers into the same repositories.
  // Does not record any repository data, only the number of times two people
  // overlapped within any repository
  var allCommits = {}
  var connectionCounts = {}

  for (var project in contributors) {
    var commitsForProject = {}
    for (var email in contributors[project]) {
      if (!commitsForProject[email]) {
        commitsForProject[email] = 0
        allCommits[email] = 0
      }
      allCommits[email] += contributors[project][email]
      commitsForProject[email] += contributors[project][email]
    }
    // Do a quadratic calculation to find the matrix overlap of all
    // contributors to this project
    for (var email1 in commitsForProject) {
      for (var email2 in commitsForProject) {
        if (email1 != email2) {
          var key = emailPairNormalized(email1, email2)
          if (!connectionCounts[key]) {
            connectionCounts[key] = 0
          }
          // The connection between two people is the minimum of their
          // contributions to a single project, calculated for each project and
          // summed
          connectionCounts[key] += Math.abs(commitsForProject[email1] - commitsForProject[email2])
        }
      }
    }
  }
  var graph = {
    nodes: [],
    links: []
  }
  for (var email in allCommits) {
    var node = {
      id: email,
      group: groupFromEmailDomain(email),
      commits: allCommits[email],
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
})
