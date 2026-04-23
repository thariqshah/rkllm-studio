#!/bin/bash

# Native Startup Script for RKLLama Studio
# Targeted for Rockchip RK3588 (FriendlyElec CM3588)

# 1. Environment Setup
export LIBRARY_PATH=/ai-shit/docker-runtime:/usr/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/ai-shit/docker-runtime:/usr/lib:$LD_LIBRARY_PATH
export RUST_LOG=info

PROJECT_ROOT=$(pwd)

echo "--- Starting RKLLama Studio (Native Mode) ---"

# 2. Start Backend
echo "[1/2] Starting Rust Backend..."
cd $PROJECT_ROOT/backend
cargo run --release &
BACKEND_PID=$!


echo "--- Services are running! ---"
echo "API:       http://aitana:8181"
echo "------------------------------"
echo "Press Ctrl+C to stop both services."

# 4. Handle Shutdown
trap "kill $BACKEND_PID exit" INT TERM
wait
