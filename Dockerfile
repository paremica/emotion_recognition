# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . /app

# Expose the port your app runs on
EXPOSE 8080

# Command to run your application
CMD ["uvicorn", "main1:app", "--host", "0.0.0.0", "--port", "8080"]
