#!/bin/bash

# Native Startup Script for RKLLama Studio
# Targeted for Rockchip RK3588 (FriendlyElec CM3588)

# 1. Environment Setup
export LIBRARY_PATH=/usr/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
export RUST_LOG=info

PROJECT_ROOT=$(pwd)

echo "--- Starting RKLLama Studio (Native Mode) ---"

# 2. Start Backend
echo "[1/2] Starting Rust Backend..."
cd $PROJECT_ROOT/backend
cargo run --release &
BACKEND_PID=$!

# 3. Start Frontend
echo "[2/2] Starting React Frontend..."
cd $PROJECT_ROOT/frontend
# Ensure dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies first..."
    npm install
fi
npm run dev -- --host &
FRONTEND_PID=$!

echo "--- Services are running! ---"
echo "Dashboard: http://aitana:5173"
echo "API:       http://aitana:8080"
echo "------------------------------"
echo "Press Ctrl+C to stop both services."

# 4. Handle Shutdown
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait
