# Gunakan image Python resmi
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Salin semua file
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port yang akan didengar (Cloud Run default: 8080)
EXPOSE 8080

# Jalankan Flask
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--threads", "2", "app:app"]