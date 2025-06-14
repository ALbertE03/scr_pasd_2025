# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages needed for Ray dashboard
RUN apt-get update && apt-get install -y procps net-tools && apt-get clean

# Copy the current directory contents into the container
COPY . .

# Expose Ray Dashboard port, Ray head port, and GCS port
EXPOSE 8265 10001 6379

CMD [ "python","train.py" ]