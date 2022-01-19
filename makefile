SRC_CU   = $(wildcard *.cu */*.cu */*/*.cu */*/*/*.cu)
SRC_CXX  = $(wildcard *.cpp */*.cpp */*/*.cpp */*/*/*.cpp)
NVCC     = nvcc
LIBS	 = -lcublas -lcusparse

FOLDER   = bin/
ROOT     = ./

NAME     = NNLib
EXE		 = $(ROOT)$(FOLDER)$(NAME)

FLAGS    = -dlto -O3 -Xptxas -O3,-v -use_fast_math

ifeq ($(OS),Windows_NT)
    SUFFIX := .exe
else
    SUFFIX :=
endif

native:
	mkdir -p $(ROOT)$(FOLDER)
	$(NVCC) $(SRC_CU) $(SRC_CXX) $(FLAGS) $(LIBS) -O3 -o $(EXE)$(SUFFIX)
