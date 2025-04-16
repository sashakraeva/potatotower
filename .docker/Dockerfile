# Use ROS Noetic as base
ARG ROS_DISTRO=noetic
FROM ros:$ROS_DISTRO-ros-base

ARG ROS_DISTRO
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

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
    terminator \
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
WORKDIR /
RUN mkdir -p potato_ws/src

COPY . /potato_ws/src/

WORKDIR /potato_ws
RUN ["/bin/bash", "-c", "source /opt/ros/$ROS_DISTRO/setup.bash &&\
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release &&\
    catkin build \
    "]

RUN ["/bin/bash", "-c", "cp /potato_ws/src/.docker/entrypoint.sh /entrypoint.sh && chmod 777 /entrypoint.sh"]
RUN ["/bin/bash", "-c", "cp /potato_ws/src/.docker/setup.bash /potato_ws/setup.bash && chmod 777 /potato_ws/setup.bash"]
ENTRYPOINT ["bash", "/entrypoint.sh" ]

CMD ["bash"]