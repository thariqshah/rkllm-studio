# --- Build Stage ---
FROM rust:slim-bookworm AS builder

# Install build dependencies for Debian
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    clang \
    libclang-dev \
    gcc-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/rkllama
COPY . .

# Verify the library is present
RUN ls -la librkllmrt.so || (echo "Library not found in backend/ folder!" && exit 1)

# Build the application (using native glibc)
ENV LIBRARY_PATH=/usr/src/rkllama
RUN cargo build --release

# --- Runtime Stage ---
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary and the RKLLM library
COPY --from=builder /usr/src/rkllama/target/release/rkllama-backend /app/rkllama-backend
COPY --from=builder /usr/src/rkllama/librkllmrt.so /usr/lib/librkllmrt.so

RUN mkdir /app/models

EXPOSE 8080
CMD ["./rkllama-backend"]
