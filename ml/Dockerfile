# The prediction API in a container: numpy + FastAPI, with the release model inside.
#   docker build -t impedance-api ml
#   docker run --rm -p 8000:8000 impedance-api        # then open http://localhost:8000/docs
FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY surrogate/ surrogate/
COPY models/impedance_surrogate_v1.npz models/
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "surrogate.api:app", "--host", "0.0.0.0", "--port", "8000"]
