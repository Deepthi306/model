# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:2.6.0

# Set the working directory to /app
WORKDIR /app

RUN 


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000","--reload"]