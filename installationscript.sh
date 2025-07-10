#!/bin/bash
# Installs all dependencies for CUDA and OpenGL development

# This script is intended for Debian-based systems (like Ubuntu)

# If permission is denied, run the script with sudo or use chmod +x installationscript.sh to make it executable.

# run the script by simply typing ./installationscript.sh in the terminal

# Install system packages
echo "Installing system dependencies..."
sudo apt update && sudo apt upgrade -y

#Package for building and compiling -- includes gcc, g++, make, etc.
sudo apt install build-essential -y

#NVIDIA CUDA Toolkit for GPU acceleration - includes nvcc
sudo apt install nvidia-cuda-toolkit -y

#includes the GLUT library for OpenGL
sudo apt install freeglut3-dev -y

#most simple and basic text editor (Vim and nano suck, don't listen to anyone who says otherwise)
sudo apt install gedit -y

#but people have different preferences, so if you want to install vim and/or nano uncomment either line below
#sudo apt install vim
#sudo apt install nano

#helps to verify OpenGL installation, and verify system meets requirements & helps diagnose CUDA-GL integration issues
sudo apt install mesa-utils -y


echo "Dependencies installation completed!"
#NOTE: we are not checking for success of each command, the script could fail miserably and tell you it succeeded
#just look for bad messages, if no bad messages then everything should be good, hopefully.....





#-----------------WINE SETUP-----------------#
# Install Wine for running Windows applications if you guy want to try that in order to run certain windows software
# Great installation guide #https://cyberpanel.net/blog/install-wine-on-ubuntu
# official WineHQ guide for Ubuntu #https://wiki.winehq.org/Ubuntu

# I could write the script but it might get messy as to what to comment out and what not to

