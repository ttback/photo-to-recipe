# Use the official Python image from the Docker Hub
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the container
COPY pyproject.toml poetry.lock /app/

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry install

# Copy the rest of the application code to the container
COPY . /app

# Command to run the application
CMD ["poetry", "run", "python", "main.py"]