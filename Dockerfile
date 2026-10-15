FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r Requirements.txt
CMD ["python", "-m", "data_platform.pipeline"]
