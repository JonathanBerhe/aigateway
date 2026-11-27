# syntax=docker/dockerfile:1

# Stage 1: build the Rust core as a Linux shared library.
FROM rust:1.100.0-slim-bookworm AS core
WORKDIR /src/core
COPY core/ ./
RUN cargo build --release --locked

# Stage 2: OpenResty with the gateway code, the core library and the test mock.
# Both stages use Debian bookworm, so the library links against the same glibc.
FROM openresty/openresty:1.31.9.1-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

ENV AIGW_PREFIX=/opt/aigw \
    AIGW_CORE_LIB=/opt/aigw/lib/libaigw_core.so \
    AIGW_ENV_FILE=/dev/null

COPY --from=core /src/core/target/release/libaigw_core.so /opt/aigw/lib/
COPY gateway/conf/nginx.conf /opt/aigw/conf/nginx.conf
COPY gateway/lua/ /opt/aigw/lua/
COPY scripts/gateway.sh /opt/aigw/bin/gateway.sh
COPY tests/mock/ /opt/mock/

# Send the NGINX logs of both servers to the container output.
RUN mkdir -p /opt/aigw/logs /opt/mock/logs \
 && ln -sf /dev/stdout /opt/aigw/logs/access.log \
 && ln -sf /dev/stderr /opt/aigw/logs/error.log \
 && ln -sf /dev/stdout /opt/mock/logs/access.log \
 && ln -sf /dev/stderr /opt/mock/logs/error.log

EXPOSE 8000 8001
ENTRYPOINT ["/opt/aigw/bin/gateway.sh"]
CMD ["start"]
