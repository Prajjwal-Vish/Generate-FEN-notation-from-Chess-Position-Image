# Use Python 3.10
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements first to cache dependencies
COPY requirements.txt requirements.txt

# Install dependencies
# We upgrade pip to avoid errors
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user (Security requirement for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the port for Hugging Face (It expects 7860)
ENV PORT=7860

# Run the application with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", "--timeout", "0"]