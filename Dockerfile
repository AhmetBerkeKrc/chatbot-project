FROM python:3.11-slim
# Set build-time variables (only available during `docker build`)
ARG GOOGLE_SHEETS_ID
ARG OPENAI_API_KEY
ARG SENDER_EMAIL
ARG SENDER_PASSWORD
ARG SMTP_PORT
ARG SMTP_SERVER

# Set runtime environment variables (available inside the running container)
ENV GOOGLE_SHEETS_ID=${GOOGLE_SHEETS_ID}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV SENDER_EMAIL=${SENDER_EMAIL}
ENV SENDER_PASSWORD=${SENDER_PASSWORD}
ENV SMTP_PORT=${SMTP_PORT}
ENV SMTP_SERVER=${SMTP_SERVER}

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
