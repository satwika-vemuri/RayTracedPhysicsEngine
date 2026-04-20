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


# to see video, cd into frames folder and type the following command:
# ffmpeg -framerate 15 -i image%d.ppm -c:v libx264 -vf "format=yuv420p" output.mp4
# 15 means 15 frames per second
# image%d.ppm indicates that the video will render frames in order of image1.ppm, image2.ppm, etc.


# Instructions to get cuda working:
# module purge
# module load legacy/CentOS7
# module load gcc/8.5.0 
# module load nvidia-hpc-sdk

# submit job "sbatch job.sl" to run with cuda


NVCC = nvcc

# Host + device flags
NVCCFLAGS = -std=c++14 -O2 -Xcompiler "-Wall -Wextra -pedantic"

TARGET = render
SRC = main.cu rayTrace.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) frames/*

.PHONY: all run clean