SRC      = $(wildcard *.cu */*.cu */*/*.cu */*/*/*.cu)
NVCC     = nvcc
LIBS	 = -lcublas -lcusparse

FOLDER   = bin/
ROOT     = ./

NAME     = NNLib
EXE		 = $(ROOT)$(FOLDER)$(NAME)

ifeq ($(OS),Windows_NT)
    SUFFIX := .exe
else
    SUFFIX :=
endif

native:
	mkdir -p $(ROOT)$(FOLDER)
	$(NVCC) $(SRC) $(LIBS) -O3 -o $(EXE)$(SUFFIX)
