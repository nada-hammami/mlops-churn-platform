FROM python:3.11-slim
WORKDIR /app
ENV GIT_PYTHON_REFRESH=quiet
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
EXPOSE 8000
CMD sh -c "python -m src.train && python -m src.promote 1 && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"