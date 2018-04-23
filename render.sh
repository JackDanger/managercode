#!/bin/bash
#
# Accepts JSON graph data through a filename argument.
# This data is placed in a new directory and the necessary html is copied in.

set -euo pipefail

set -x
function main() {
  set -x
  echo ${*}
  local name=$1
  local filename=$2
  if [[ -z $name ]] || [[ -z "${filename}" ]]; then
    1>&2 echo "USAGE: ${0} name-of-graph /path/to/graphData.json"
    exit 1
  fi

  local parameterized_name=$(echo ${name} | tr -cd '[[:alpha:]]- ' | tr ' ' '-')

  local deploy_dir=${parameterized_name}.gen

  # make a copy of all assets into this new subdirectory
  mkdir -p $deploy_dir
  cp ${filename} ${deploy_dir}/graphData.json
  cp index.html.template ${deploy_dir}/index.html
  cp force-graph.js ${deploy_dir}/
  cp adjacency-matrix.js ${deploy_dir}/
  cp d3*.js ${deploy_dir}/

  # Start the server
  if nc -z localhost 5050; then
    echo "Server is already running"
  else
    echo "Starting webserver"
    python3 -m http.server 5050 &
    sleep 2
  fi
  open http://localhost:5050/${deploy_dir}/index.html
}

main "${@}"
