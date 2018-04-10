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

  local parameterized_name=$(parameterize "${name}")

  local tmpdir=${parameterized_name}.gen
  mkdir $tmpdir
  cp ${filename} ${tmpdir}/graphData.json
  cp index.html.template ${tmpdir}/index.html
  cp force-graph.js ${tmpdir}/
  cp adjacency-matrix.js ${tmpdir}/
  cp d3*.js ${tmpdir}/

  pushd $tmpdir
  python3 -m http.server 5050 &
  sleep 2
  open http://localhost:5050/index.html
  popd
}

parameterize() {
  set -x
  echo $@ | tr -cd '[[:alpha:]] ' | tr ' ' '-'
}
export -f parameterize

main "${@}"
