FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install with pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train2.py predict.py server.py ./
COPY model_xgb.bin ./

EXPOSE 9696

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9696"]
