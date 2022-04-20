#OBJS specifies which files to compile as part of the project
OBJS = src/cool.cu

#CC specifies which compiler we're using
CC = nvcc

#COMPILER_FLAGS specifies the additional compilation options we're using
# -w suppresses all warnings
COMPILER_FLAGS = -w -g -O3 -I./include

#LINKER_FLAGS specifies the libraries we're linking against
LINKER_FLAGS = -lSDL2

#OBJ_NAME specifies the name of our exectuable
OBJ_NAME = cool

#This is the target that compiles our executable
all : $(OBJS)
	$(CC) $(OBJS) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $(OBJ_NAME)

test: src/test.cpp
	g++ src/test.cpp $(COMPILER_FLAGS) $(LINKER_FLAGS) -o test
