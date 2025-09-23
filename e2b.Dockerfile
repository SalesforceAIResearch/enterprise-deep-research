# You can use most Debian-based base images
# FROM ubuntu:22.04
FROM e2bdev/code-interpreter:latest 
# Install dependencies and customize sandbox

# Install some Python packages
RUN pip install cowsay 

