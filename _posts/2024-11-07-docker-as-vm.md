---
title: Using Docker as Virtual Machine for Deep Learning
categories:
  - blog
tags:
  - en
  - tutorial
toc: True
---
So you don't have to deal with Windows environment setup

## Why?

I personally have suffered enough from setting up environment on Windows that I have given up the idea on using Windows for CUDA-related projects.
I have tried WSL2 with conda managing the environment, but it always haunts me with some weird issues.

Of course I still use WSL2 for other purposes, but for deep learning, I prefer just to use docker. 
Yes, I am using docker like a virtual machine. 
It is not the most efficient way, but it is the most convenient way for me. 

## Setting up Docker

Here is a simple `Dockerfile` that you can use to create a container for this project:

```Dockerfile
# Use NVIDIA PyTorch base image
# If you run into issues, try using a different version of the base image
FROM nvcr.io/nvidia/pytorch:24.07-py3 

# Set the working directory in the container
WORKDIR /workspace

# Copy current directory contents into the container's /workspace directory
COPY . /workspace

# Install other packages as needed
RUN pip install jupyter

# Expose Jupyter Notebook default port
EXPOSE 8888

# Set the default command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

Here is the `docker-compose.yml` file that you can use to start the container:

```yaml
services:
  pytorch_app:
    build:
      context: .
      dockerfile: Dockerfile
    image: pytorch_app_image
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all  # Use all available GPUs
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility  # Necessary driver capabilities, it could also include `video`
    ports:
      - "8888:8888"  # Map Jupyter Notebook port to host
    volumes:
      - .:/workspace  # Mount current directory to /workspace in the container
```

This will install CUDA and cuDNN, as well as PyTorch.
Building on top of this image, you can install other libraries as needed.

You might also need to alter some docker daemon settings to allow the container to access the GPU.

```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

## Running the Container

To build the container, run the following command in the same directory as the `Dockerfile`:

```bash
docker compose up
```

This will build the container and start it. You can access the Jupyter Notebook server by navigating to `http://localhost:8888` in your browser.
