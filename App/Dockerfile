# Stage 1: Build stage
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Copy the project files to the working directory
COPY app.py /app
COPY requirements.txt /app
COPY model_downloader.py /app
COPY model-builder /app/model-builder
COPY static /app/static
COPY templates /app/templates

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the model downloader to check and download the model files
RUN python model_downloader.py --download

# Stage 2: Final stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the built dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Install gunicorn
RUN pip install --no-cache-dir gunicorn

# Copy the project files to the working directory
COPY app.py /app
COPY custom-model /app/custom-model
COPY static /app/static
COPY templates /app/templates

# Expose the port
EXPOSE 5000

# Set the entrypoint command to run the Flask app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
