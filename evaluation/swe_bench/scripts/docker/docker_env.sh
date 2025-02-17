#!/bin/bash

# Set Docker to run non-interactively
export DOCKER_BUILDKIT=1
export DOCKER_SCAN_SUGGEST=false
export DEBIAN_FRONTEND=noninteractive

# Function to run Docker commands with yes piped in
docker_noninteractive() {
    yes | "$@"
}

# Alias docker to use the non-interactive function
alias docker=docker_noninteractive