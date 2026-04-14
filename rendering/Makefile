

# Thnx Claude

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