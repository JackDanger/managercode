#!/bin/bash

&>/dev/null which jq || brew install jq

clone_org() {
  # Clones all public repos for a github organization
  # updates them if they are already cloned
  local organization=$1
  local checkout_location=$2
  curl -s "https://api.github.com/orgs/${organization}/repos?per_page=1000" |
    jq '.[] | .full_name' |
    sed 's/"//g' |
    while read repo; do
      # ${repo} ~= mapbox/Simple-KML
      if [[ -d /${repo} ]]; then
        echo "Updating ${repo}"
        cd ${checkout_location}/${repo}/
        git pull --recurse-submodules
      else
        echo "Cloning ${repo}"
        git clone --recursive https://github.com/${repo}.git ${checkout_location}/${repo}
      fi
  done
}

recent_work() {
  local checkout_location=$1
  find ${checkout_location} -type d -depth 1 | while read repo; do
    cd ${repo}
    git log -n 1 --format="%ct $(echo $repo) updated  %ar by %an"
    cd ${checkout_location}
  done |
    sort -n |
    awk '{$1=""; print $0}'  # print all but the first column
}

usage() {
  echo "Usage:"
  echo "$0 clone_org some_github_org /repo/path"
  echo "$0 recent_work /repo/path"
  exit 1
}

case $1 in
  "clone_org")
    [[ -z "$3" ]] && usage
    clone_org $2 $3
    ;;
  "recent_work")
    [[ -z "$2" ]] && usage
    recent_work $2
    ;;
  "")
    usage
    ;;
  esac
