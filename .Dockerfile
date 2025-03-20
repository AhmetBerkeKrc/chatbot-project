FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the application port (modify as needed)
EXPOSE 8000

# Set the entrypoint command (modify as needed)
CMD ["uvicorn", "healthie_backend:app", "--host", "0.0.0.0", "--port", "8000"]
