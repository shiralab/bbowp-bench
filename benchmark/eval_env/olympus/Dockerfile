FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]