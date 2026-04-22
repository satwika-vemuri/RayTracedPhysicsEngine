# Thnx Claude

#####################
#---INSTRUCTIONS---
#####################
# Build both targets:       make
# Build parallel only:      make parallel
# Build serial only:        make serial
# Run parallel:             make run_parallel
# Run serial:               make run_serial
# Clean binaries + frames:  make clean

# To stitch frames into video:
# ffmpeg -framerate 15 -i frames/image%d.ppm -c:v libx264 -vf "format=yuv420p" output.mp4

# To get CUDA working on the cluster:
# module purge
# module load legacy/CentOS7
# module load gcc/8.5.0
# module load nvidia-hpc-sdk
# Submit job: sbatch job.sl


NVCC     = nvcc
CXX      = g++
INCLUDES = -Iinclude

NVCCFLAGS = -std=c++14 -O2 -Xcompiler "-Wall -Wextra -pedantic" $(INCLUDES)
CXXFLAGS  = -std=c++17 -O2 -Wall -Wextra $(INCLUDES)

PARALLEL_TARGET = render_parallel
SERIAL_TARGET   = render_serial

PARALLEL_SRC = src/parallel/main.cu src/parallel/rayTrace.cu
SERIAL_SRC   = src/serial/main.cpp

all: $(PARALLEL_TARGET) $(SERIAL_TARGET)

parallel: $(PARALLEL_TARGET)

serial: $(SERIAL_TARGET)

$(PARALLEL_TARGET): $(PARALLEL_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $(PARALLEL_SRC)

$(SERIAL_TARGET): $(SERIAL_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SERIAL_SRC)

run_parallel: $(PARALLEL_TARGET)
	./$(PARALLEL_TARGET)

run_serial: $(SERIAL_TARGET)
	./$(SERIAL_TARGET)

clean:
	rm -f $(PARALLEL_TARGET) $(SERIAL_TARGET) frames/*

.PHONY: all parallel serial run_parallel run_serial clean
