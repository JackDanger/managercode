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
    console.log(`Usage: ${__filename} path/to/directory`);
    process.exit(1);
}
// cd /first/argument
var repo_list = process.argv[2]


var contributors = {}
function addContribution(project, email) {
  // Record a person contributing a commit to a repo
  // Use the basedir name of the project as the key
  var parts = project.split('/')
  projectName = parts[parts.length - 1]
  if (!contributors[projectName]) {
    contributors[projectName] = {}
  }
  if (!contributors[projectName][email]) {
    contributors[projectName][email] = 1
  } else {
    contributors[projectName][email]++
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

function contributorWeightedGraph() {
  // Calculate the connections between committers into the same repositories.
  // Does not record any repository data, only the number of times two people
  // overlapped within any repository
  var allCommits = {}
  var connectionCounts = {}

  for (var project in contributors) {
    for (var email in contributors[project]) {
      if (!allCommits[email]) {
        allCommits[email] = 0
      }
      allCommits[email] += contributors[project][email]
    }
    // Do a quadratic calculation to find the matrix overlap of all
    // contributors to this project
    for (var email1 in contributors[project]) {
      for (var email2 in contributors[project]) {
        if (email1 != email2) {
          var key = emailPairNormalized(email1, email2)
          if (!connectionCounts[key]) {
            connectionCounts[key] = 0
          }
          // The connection between two people is the minimum of their
          // contributions to a single project, calculated for each project and
          // summed
          connectionCounts[key] += Math.min(contributors[project][email1], contributors[project][email2])
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
      count: allCommits[email],
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

function repoWeightedGraph() {
  // Calculate the connections between repos by how many authors they share.
  var allCommits = {}
  var connectionCounts = {}

  for (var project in contributors) {
    for (var email in contributors[project]) {
      if (!allCommits[project]) {
        allCommits[project] = 0
      }
      allCommits[project] += contributors[project][email]
    }
    // Do a quadratic calculation to find the number of shared
    // contributors this repo has.
    for (var project1 in contributors) {
      for (var project2 in contributors) {
        if (project1 != project2) {
          var key = [project1, project2].join(',')
          for (var email in contributors[project1]) {
            if (contributors[project2][email]) {
              if (!connectionCounts[key]) {
                connectionCounts[key] = 0
              }
              // Weight the link by the minimum number of commits each
              // person has put into _both_ repos.
              connectionCounts[key] += Math.min(contributors[project1][email],
                                                contributors[project2][email])
            }
          }
        }
      }
    }
  }
  var graph = {
    nodes: [],
    links: []
  }
  for (var project in allCommits) {
    var node = {
      id: project,
      group: 1,
      count: allCommits[project],
    }
    graph.nodes.push(node)
  }
  for (var repoPairKey in connectionCounts) {
    var repoPair = repoPairKey.split(',')
    var link = {
      source: repoPair[0],
      target: repoPair[1],
      value: connectionCounts[repoPair]
    }
    graph.links.push(link)
  }
  return graph
}

glob(repo_list + "/*/.git", function(err, files) {
  for (var idx in files) {
    var basedir = files[idx].replace('/.git', '')
    console.log("repo: " + basedir)
    var cmd = "git log --format='%ae' --since='3 months ago'"
    try {
      var buff = execSync(cmd, {cwd: basedir})
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
    catch(e) {
      console.error(`Command failed in ${basedir}: ${cmd}`)
    }
  }
  fs.writeFile(__dirname + "/contributorGraphData.json", JSON.stringify(contributorWeightedGraph(), null, 2), function(err) {
    if(err) {
        return console.log(err);
    }
  }); 
  fs.writeFile(__dirname + "/repoGraphData.json", JSON.stringify(repoWeightedGraph(), null, 2), function(err) {
    if(err) {
        return console.log(err);
    }
  }); 
  console.log("The contribution graph has been regenerated into ./" + __dirname + "/*GraphData.js")
  console.log("Run a web server and open the html page to view results")
  console.log("python -m SimpleHTTPServer & (sleep 5 && open http://localhost:8000/" + __dirname + "/index.html) && fg")
})
