#!/bin/bash
set -euf -o pipefail

# Installs poetry
curl -sSL https://install.python-poetry.org | python3 -

# Creates Python virtual environment using Poetry
poetry install --no-root
poetry shell
