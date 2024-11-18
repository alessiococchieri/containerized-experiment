FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install python and general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
    git \
    bash \
    nano \
    wget \
    python3.10 \
    python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /containerized-experiment
VOLUME "/data"
ENV DATA_DIR=/data
WORKDIR /containerized-experiment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY * /containerized-experiment/
ENV OWNER=1157:1157
CMD export OUTPUT_DIR=$DATA_DIR/$(date +%Y-%m-%d-%H-%M-%S)-$(hostname) && \
    mkdir -p $OUTPUT_DIR && \
    python3 bench.py | tee $OUTPUT_DIR/output.log && \
    chown -R $OWNER $DATA_DIR