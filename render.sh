#!/bin/bash
#
# Accepts JSON graph data through a filename argument.
# This data is placed in a new directory and the necessary html is copied in.

set -eo pipefail

function main() {
  echo ${*}
  local name=$1
  local filename=$2
  local s3=$3
  if [[ -z $name ]] || [[ -z "${filename}" ]]; then
    1>&2 echo "USAGE: ${0} name-of-graph /path/to/graphData.json [S3_PATH]"
    exit 1
  fi

  set -x
  local parameterized_name=$(echo ${name} | tr -cd '[[:alpha:]0-9]- ' | tr ' ' '-')

  local deploy_dir=${parameterized_name}.gen

  # make a copy of all assets into this new subdirectory
  mkdir -p $deploy_dir
  cp ${filename} ${deploy_dir}/graphData.json
  cp index.html ${deploy_dir}/index.html

  if [[ -n "${s3}" ]]; then
    if [[ "${s3}" == "--s3" ]]; then
      deploy_to_s3 ${parameterized_name} "s3://jackdanger.com/managercode/${parameterized_name}" 
    else
      deploy_to_s3 ${parameterized_name} ${s3}
    fi
  else
    render_and_serve "${parameterized_name}" "${filename}"
  fi
}

deploy_to_s3() {
  set -x
  local parameterized_name=$1
  local s3_path=$2
  aws s3 cp --recursive ${parameterized_name}.gen ${s3_path}  --acl public-read
  echo "open ${s3_path}/index.html"
}

render_and_serve() {
  local parameterized_name=$1
  local filename=$2

  # Start the server
  if nc -z localhost 5050; then
    echo "Server is already running"
  else
    echo "Starting webserver"
    python3 -m http.server 5050 &
    sleep 2
  fi
  open http://localhost:5050/${deploy_dir}/index.html
  wait
}

main "${@}"
