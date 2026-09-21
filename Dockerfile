FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements-data-platform.txt
CMD ["python", "-m", "data_platform.pipeline"]
