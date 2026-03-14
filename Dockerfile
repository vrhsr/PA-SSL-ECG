# Use the official NVIDIA NGC PyTorch image as the base
# This comes pre-configured with CUDA, cuDNN, PyTorch, and optimizations
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set the working directory in the container
WORKDIR /workspace/PA-SSL-ECG

# We do not copy the code or data into the image during build.
# The college instructions prefer pulling an image and running it,
# so we will mount the codebase and data at runtime to ensure persistence.

# Copy ONLY the requirements to install dependencies
COPY requirements.txt /tmp/requirements.txt

# Install the Python dependencies (PyTorch is already installed)
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir umap-learn gdown

# Default command just keeps the container alive if run interactively
CMD ["/bin/bash"]
