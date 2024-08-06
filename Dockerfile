# Use the official PyTorch image with CUDA and cuDNN support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the image
# COPY . .

# Expose port 8000 (or any port your application needs)
# EXPOSE 8000

# Run the application (replace 'app.py' with your main application script)
# CMD ["python", "app.py"]
