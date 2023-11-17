# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim
# Set working directory
WORKDIR /usr/src/app
# Copy
COPY test.py .
# run
CMD [ "python", "./test.py" ]
