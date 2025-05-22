FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for model registry
RUN mkdir -p model_registry

# Expose the port for the API
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "server.model_server"]