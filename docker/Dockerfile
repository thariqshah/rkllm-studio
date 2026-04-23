# --- Build Stage ---
FROM rust:1.77-slim-bookworm AS builder

WORKDIR /usr/src/rkllama
COPY . .

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Build the application
# Note: In a real Rockchip environment, you would need to cross-compile or build on-device
RUN cargo build --release

# --- Runtime Stage ---
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /usr/src/rkllama/target/release/rkllama-backend /app/rkllama-backend

# Create models directory
RUN mkdir /app/models

# Expose the API port
EXPOSE 8080

# Run the application
CMD ["./rkllama-backend"]
