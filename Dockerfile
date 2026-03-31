# 1. Base Image - Use Python 3.10 slim for a smaller footprint
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Set Environment Variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1
# Set cache folder for pix2tex to use inside the container
ENV XDG_CACHE_HOME=/app/.cache 

# 4. Install system dependencies required by Pillow and ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
# Copy only requirements first, to cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Pre-download Model Weights (MLOps Best Practice)
# By running latexocr once during the build process, the Docker image will cache the downloaded PyTorch weights.
# This prevents the server from having to download hundreds of MBs during container startup in production.
RUN python -c "from pix2tex.cli import LatexOCR; LatexOCR()"

# 7. Copy the rest of the application code
COPY ./app ./app

# 8. Expose the FastAPI port
EXPOSE 8000

# 9. Define the start up command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
