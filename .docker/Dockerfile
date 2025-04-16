# Use ROS Noetic as base
ARG ROS_DISTRO=noetic
FROM ros:$ROS_DISTRO-ros-base

ARG ROS_DISTRO
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM xterm-256color

# Basic tools and Python setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    python3-pip \
    python3-venv \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall-generator \
    ros-$ROS_DISTRO-rviz \
    ros-$ROS_DISTRO-rqt \
    ros-$ROS_DISTRO-rqt-common-plugins \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python project dependencies
RUN python3 -m pip install \
    pygame \
    numpy \
    shapely \
    svgpathtools \
    stable-baselines3 \
    gymnasium \
    matplotlib \
    pandas \
    pymunk

# Create and build catkin workspace
WORKDIR /dev_ws
RUN mkdir -p src
COPY . /dev_ws/src/

# Source ROS and build the workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd /dev_ws && \
    catkin config --extend /opt/ros/$ROS_DISTRO && \
    catkin build"

# Copy and make entrypoint scripts executable
RUN cp /dev_ws/src/.docker/entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh
RUN cp /dev_ws/src/.docker/setup.bash /dev_ws/setup.bash && chmod +x /dev_ws/setup.bash

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
