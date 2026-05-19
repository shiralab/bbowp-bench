FROM python:3.11-slim

WORKDIR /app

ENV MUJOCO_GL=egl \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-dri \
    libegl1 \
    libgles2 \
    libosmesa6 \
    libglfw3 \
    patchelf \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]