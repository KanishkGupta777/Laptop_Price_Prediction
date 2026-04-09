# ─────────────────────────────────────────────────────────────
#  Dockerfile — Laptop Price Prediction API
# ─────────────────────────────────────────────────────────────
#  Build:  docker build -t laptop-price-api .
#  Run:    docker run -p 5000:5000 laptop-price-api
#  Test:   curl http://localhost:5000
# ─────────────────────────────────────────────────────────────

# Base image: slim Python 3.10
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies first
# (Docker caches this layer — only re-runs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port 5000 for the Flask API
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=project_root/api/app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run the API when container starts
CMD ["python", "project_root/api/app.py"]
