# Multi-stage Dockerfile for BYO-EVAL project
# Stage 1: Base image with Blender and system dependencies
FROM ubuntu:22.04 as blender-base

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Blender dependencies
    libx11-6 \
    libxrender1 \
    libxext6 \
    libxi6 \
    libxrandr2 \
    libxfixes3 \
    libxcursor1 \
    libxinerama1 \
    libfreetype6 \
    libfontconfig1 \
    libxss1 \
    libglu1-mesa \
    libegl1-mesa \
    # Additional X11 and GUI libraries needed by Blender
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    libdbus-1-3 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libcairo-gobject2 \
    libpango-1.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    # Virtual display support
    xvfb \
    x11-utils \
    # Python and build tools
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    # Additional utilities
    wget \
    curl \
    unzip \
    git \
    # Process management
    procps \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install Blender 4.0
RUN wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz \
    && tar -xf blender-4.0.2-linux-x64.tar.xz \
    && mv blender-4.0.2-linux-x64 /opt/blender \
    && rm blender-4.0.2-linux-x64.tar.xz

# Add Blender to PATH
ENV PATH="/opt/blender:$PATH"

# Create symlink for easier access
RUN ln -s /opt/blender/blender /usr/local/bin/blender

# Stage 2: Application setup
FROM blender-base as app-base

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt pyproject.toml ./

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# Install the project dependencies
RUN python -m pip install -r requirements.txt

# Install additional dependencies that might be needed
RUN python -m pip install \
    pillow \
    scipy \
    tqdm \
    requests \
    typing-extensions

# Stage 3: Final application
FROM app-base as final

# Copy the entire project
COPY . .

# Install the project in development mode
RUN python -m pip install -e .

# Create directories for input/output
RUN mkdir -p /app/input /app/output /app/data /app/cache

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV BLENDER_USER_SCRIPTS="/app/scripts"
ENV DISPLAY=:99
# Additional environment variables for headless operation
ENV XDG_RUNTIME_DIR="/tmp"
ENV WAYLAND_DISPLAY=""
ENV GDK_BACKEND="x11"

# Copy and setup entry point scripts
COPY docker/*.sh /app/scripts/
RUN chmod +x /app/scripts/*.sh

# Set the main entrypoint
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Default command shows help
CMD ["help"]

# Expose common ports (if needed for web interfaces)
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import bpy; print('Blender Python API available')" || exit 1 