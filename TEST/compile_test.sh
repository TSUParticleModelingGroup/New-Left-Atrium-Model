# Compiles and runs the program

#if you get a permission denied error, type this command in the terminal:
#chmod 755 compile.sh


#We should be in the Source in the test directory
#compile the ImGui files

nvcc -c imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp
nvcc -c imgui/imgui_impl_glfw.cpp imgui/imgui_impl_opengl3.cpp


nvcc SVT.cu glad.c\
     imgui.o imgui_draw.o imgui_tables.o imgui_widgets.o imgui_impl_glfw.o imgui_impl_opengl3.o \
     -o svt \
     -lglfw -lGL -lGLU -lm -lX11 -lXrandr -lXinerama -lXcursor -lXi

#move svt and all .o files to ExecutableFiles
mv svt bin/

mv *.o bin/
