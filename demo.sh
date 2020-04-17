#!/usr/bin/env bash
set -e

die () {
	echo >&2 "$@"
	exit 1
}

[ -x "$(command -v virtualenv)" ] || die 'Error: virtualenv is not installed. Please run apt install python-virtualenv'

REPO_ROOT=$(cd $(dirname "$0"); pwd)
ARGUMENTS=$@

TOOL_PATH="${REPO_ROOT}/src/main.py"
ENV_PATH="${REPO_ROOT}/env"
PYTHON_BIN="${ENV_PATH}/bin/python3"
PIP_BIN="${ENV_PATH}/bin/pip3"

# cd ${REPO_ROOT}

# Ensure we have a python virtual environment with all depenendcies
if [ ! -d "${ENV_PATH}" ]; then
	echo "o Initial setup"
	virtualenv -p python3 ${ENV_PATH}
	echo
	echo "o Installing dependencies"
	$PIP_BIN install -r "${REPO_ROOT}/requirements.txt"
	echo
fi

cd ${REPO_ROOT}
$PYTHON_BIN $TOOL_PATH $ARGUMENTS
