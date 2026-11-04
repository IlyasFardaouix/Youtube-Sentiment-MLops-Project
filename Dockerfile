FROM python:3.10-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY app_api.py .
COPY src ./src
COPY models ./models
EXPOSE 7860
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]
