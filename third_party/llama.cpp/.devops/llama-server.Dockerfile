ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION as build

RUN apt-get update && \
    apt-get install -y build-essential git libcurl4-openssl-dev

WORKDIR /app

COPY . .

ENV LLAMA_CURL=1

RUN make -j$(nproc) llama-server

FROM ubuntu:$UBUNTU_VERSION as runtime

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libgomp1

COPY --from=build /app/llama-server /llama-server

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/llama-server" ]
