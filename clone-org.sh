#!/bin/bash

&>/dev/null which jq || brew install jq

clone_org() {
  set -x
  # Clones all public repos for a github organization
  # updates them if they are already cloned
  local organization=$1
  local checkout_location=$2

  # Create the directory if it doesn't exist
  mkdir -p "$checkout_location"

  # Change to the specified directory
  cd "$checkout_location" || exit

  # Fetch the list of repositories for the organization and clone each one
  gh repo list "$organization" --limit 1000 --json name -q '.[].name' | while read -r repo; do
    git_dir="${checkout_location}/${repo}"
    if [[ -d $git_dir ]]; then
      true
      #pushd $git_dir
      #git pull --recurse-submodules
      #popd
    else
      gh repo clone "$organization/$repo"
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
  echo "$0 SOME_GITHUB_organization /repo/path"
  echo "$0 'recent' /repo/path"
  exit 1
}

case $1 in
  "recent")
    [[ -z "$2" ]] && usage
    recent_work $2
    exit 0
    ;;
  "")
    usage
    ;;
  *)
    [[ -z "$2" ]] && usage
    clone_org $1 $2
    exit 0
    ;;
  esac
usage
