FROM unibo-img:latest
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
    CUDA_VISIBLE_DEVICES=0 python3 bench.py | tee $OUTPUT_DIR/output.log && \
    chown -R $OWNER $DATA_DIR