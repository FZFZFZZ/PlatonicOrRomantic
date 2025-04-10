FROM python:3.10

WORKDIR /app

RUN pip install uv

COPY requirements.txt .

RUN uv venv && uv pip install --no-cache-dir -r requirements.txt

COPY preprocess*.py .
COPY models/*.py models/
# Copy the only model needed to run the server
COPY models/lstm_basic_0.glove.6B.50d.pth models/
COPY vectors/glove.6B.50d.txt vectors/
COPY server/*.py server/

CMD ["uv", "run", "uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
