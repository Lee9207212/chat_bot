FROM python:3.11-slim

WORKDIR /app

# System dependencies for WebRTC (libnice, gstreamer etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libsrtp2-dev \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Expose ports
# 8300: HTTP signaling (FastAPI)
# 40000-40100: UDP for WebRTC ICE
EXPOSE 8300
EXPOSE 40000-40100/udp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8300/health')" || exit 1

# Run
CMD ["python", "server.py"]
