FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get update && apt-get upgrade -y
RUN apt-get install vim python3-pip cmake git -y

RUN apt-get install libgl1-mesa-glx libglx-mesa0 -y

RUN apt-get install \
    libopengl0 \
    libgl1 \
    libglx0 \
    libglvnd0 \
    libegl1 \
    libgles2 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxcb1 \
    libxcb-xkb1 \
    libx11-6 \
    libx11-xcb1 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    libglib2.0-0 \
    libglib2.0-bin \
    libgtk-3-0 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 -y

RUN apt-get install \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libglib2.0-0 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    libopengl0 \
    x11-utils -y

RUN apt-get install libnss3 -y

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get install wget ca-certificates gnupg lsb-release software-properties-common -y

SHELL ["/bin/bash", "-c"]
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN add-apt-repository -y "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
RUN apt-get install nsight-systems -y

ENV PATH="/opt/nvidia/nsight-systems/2025.5.1/bin:${PATH}"


ENV SHELL /bin/bash

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
CMD ["/bin/bash"]