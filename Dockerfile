FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl gcc g++ pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev \
        libopus-dev libvpx-dev libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY multiagent/requirements.txt multiagent/requirements.txt
COPY react_agent/requirements.txt react_agent/requirements.txt
COPY generic_agent/requirements.txt generic_agent/requirements.txt

RUN pip install --no-cache-dir \
    "pipecat-ai[silero,webrtc]>=0.0.104" \
    "langgraph-cli[inmem]>=0.4" \
    fastapi uvicorn httpx websockets aiohttp \
    sse-starlette loguru python-dotenv pydantic numpy soundfile \
    -r multiagent/requirements.txt \
    -r react_agent/requirements.txt \
    -r generic_agent/requirements.txt

COPY . .

RUN chmod +x docker/entrypoint-*.sh

EXPOSE 7860

ENTRYPOINT ["/bin/bash"]
