#!/bin/bash

&>/dev/null which jq || brew install jq

clone_org() {
  # Clones all public repos for a github organization
  # updates them if they are already cloned
  local organization=$1
  local checkout_location=$2
  if [[ -n $GITHUB_TOKEN ]]; then
    authorization="Authorization: token ${GITHUB_TOKEN}"
  fi
  curl -s "https://api.github.com/orgs/${organization}/repos?type=sources" -H "${authorization}"
  local total_pages=$(curl -s "https://api.github.com/orgs/${organization}/repos?type=sources" -H "${authorization}" -I | egrep -o 'page=\d*>; rel="last"' | cut -d = -f 2 | cut -d '>' -f 1)
  if [[ "${total_pages}" == "" ]];then
    total_pages=1;
    #1>&2 echo "Authorization required"
    #1>&2 echo "set the GITHUB_TOKEN environment variable"
    #exit 1
  fi

  for page in $(seq $total_pages); do
    echo "Cloning page ${page}"
    _clone_page "${organization}" "${checkout_location}" "${page}" "${authorization}"
  done
}

_clone_page() {
  local organization=$1
  local checkout_location=$2
  local page=$3
  local authorization=$4
  curl -s "https://api.github.com/orgs/${organization}/repos?page=${page}&type=sources" -H "${authorization}" |
    jq '.[] | .full_name' |
    sed 's/"//g' |
    while read repo; do
      if [[ -d ${checkout_location}/${repo} ]]; then
        echo "Updating ${repo}"
        cd ${checkout_location}/${repo}/
        git pull --recurse-submodules
      else
        echo "Cloning ${repo}"
        if [[ -n "${GITHUB_TOKEN}" ]]; then
          echo "token exists"
          git clone --recursive git@github.com:${repo}.git ${checkout_location}/${repo}
        else
          echo "no token"
          git clone --recursive https://github.com/${repo}.git ${checkout_location}/${repo}
        fi
      fi
    done
  # Wait until all background jobs finish
  wait
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
  echo "$0 SOME_GITHUB_ORG /repo/path"
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
