# --- Build Stage ---
FROM rust:1.77-alpine AS builder

# Install build dependencies for Alpine
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    openssl-dev \
    clang-dev \
    gcc \
    g++ \
    make

WORKDIR /usr/src/rkllama
COPY . .

# Build the application
RUN cargo build --release

# --- Runtime Stage ---
FROM alpine:latest

WORKDIR /app

# Install glibc compatibility and runtime libs for Rockchip compatibility
RUN apk add --no-cache \
    gcompat \
    libstdc++ \
    ca-certificates \
    libgcc

# Copy the binary from the builder stage
COPY --from=builder /usr/src/rkllama/target/release/rkllama-backend /app/rkllama-backend

# Create models directory
RUN mkdir /app/models

# Expose the API port
EXPOSE 8080

# Run the application
CMD ["./rkllama-backend"]
