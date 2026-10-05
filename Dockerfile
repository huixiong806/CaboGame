FROM rust:1-alpine AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY research/bin ./research/bin
COPY templates ./templates
COPY static ./static
RUN cargo build --release --bin cabo-server

FROM alpine:3.20
RUN adduser -D -H cabo
WORKDIR /app
COPY --from=build /app/target/release/cabo-server /usr/local/bin/cabo-server
COPY models ./models
USER cabo
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/cabo-server"]
