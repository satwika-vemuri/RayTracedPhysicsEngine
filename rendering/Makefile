# Thnx Claude

#####################
#---INSTRUCTIONS---
#####################
# To delete your binaries type " make clean" in your terminal.
# to compile, type "make" in ypur terminal.
# To compile and run, type "make run" in your terminal. 
# alternatively you can also type "make clean && make run" which will delete the binaries from 
# previous, compile, and then run.
# if you want to double check you're in the correct directory (currently named "rendering"), type "ls"
# inside your terminal and you should see the Makefile, alongside the other .cpp, and .h files in the output


CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O2

TARGET = render
SRC = main.cpp rayTrace.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET) > output.ppm

clean:
	rm -f $(TARGET) output.ppm

.PHONY: all run clean