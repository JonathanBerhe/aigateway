# syntax=docker/dockerfile:1
# The semantic cache sidecar. Build it from the repository root, because the
# sidecar links the Rust core (../core):
#   docker build -f semcache/Dockerfile -t aigw-semcache .

# Stage 1: build the sidecar. The Burn Flex backend is pure Rust; the
# tokenizer needs a C compiler for Oniguruma, which the Rust image has.
FROM rust:1.100.0-slim-bookworm AS build
WORKDIR /src
COPY core/ core/
COPY semcache/ semcache/
RUN cargo build --release --locked --manifest-path semcache/Cargo.toml --bin aigw-semcache

# Stage 2: a small Debian image with the binary. CA certificates let the
# sidecar download a Hub model (--model l6) over HTTPS.
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/semcache/target/release/aigw-semcache /usr/local/bin/aigw-semcache
EXPOSE 7700
ENTRYPOINT ["aigw-semcache"]
CMD ["--listen", "0.0.0.0:7700", "--model", "l6"]
