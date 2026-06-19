#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -DONNXRUNTIME_ROOT=/Users/yunusi/Documents/LLMs_dev/onnxruntime-osx-x86_64-1.22.0
cmake --build build
