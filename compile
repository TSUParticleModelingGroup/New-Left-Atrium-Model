#!/bin/bash
# filepath: c:\Users\Mason\Downloads\Code\New-Left-Atrium-Model\compile.sh

# Compiles and runs the program

#if you get a permission denied error, type this command in the terminal:
#chmod 755 compile.sh

cd Source

#Now we need to compile imGUI files too, they feel left out

nvcc -c imgui/imgui.cpp imgui/imgui_demo.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp
nvcc -c imgui/imgui_impl_glfw.cpp imgui/imgui_impl_opengl3.cpp


nvcc SVT.cu glad.c\
     imgui.o imgui_demo.o imgui_draw.o imgui_tables.o imgui_widgets.o imgui_impl_glfw.o imgui_impl_opengl3.o \
     -o svt \
     -lglfw -lGL -lGLU -lm -lpthread -ldl -lX11 -lXrandr -lXinerama -lXcursor -lXi

mv svt ../ExecutableFiles