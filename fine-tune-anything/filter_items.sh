#!/bin/bash

set -x

exec jq \
  -c \
  'select([.messages[] | select(.role == "user" and (.content | test($pattern)))] | length > 0)' \
  --arg pattern "$1" \
  $2
