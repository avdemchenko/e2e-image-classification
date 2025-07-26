# Base image with Miniconda installed
FROM continuumio/miniconda3:23.5.2-0

# Prevent interactive prompts during environment creation
ENV CONDA_NONINTERACTIVE=true

# Copy conda environment definition
COPY environment.yml /tmp/environment.yml

# Create the conda environment and clean up intermediate files
RUN conda env create -f /tmp/environment.yml && \
    conda clean -a -y

# Make the environment active by default
ENV PATH /opt/conda/envs/lightweight-cnn/bin:$PATH

# Set working directory
WORKDIR /workspace

# Copy project code into the image
COPY . /workspace

# Default command (can be overridden)
CMD ["python", "train.py"] 