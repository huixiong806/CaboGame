FROM rust:1-alpine AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY templates ./templates
COPY static ./static
RUN cargo build --release

FROM alpine:3.20
RUN adduser -D -H cabo
COPY --from=build /app/target/release/cabo-server /usr/local/bin/cabo-server
USER cabo
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/cabo-server"]
