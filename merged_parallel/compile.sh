#!/bin/bash
nvcc -ccbin /usr/bin/gcc-14 -std=c++17 -O2 main.cu SPH.cu March.cu rayTrace.cu -I. -lstdc++ -lm -o physics_sim
nvcc -ccbin /usr/bin/gcc-14 -std=c++17 -O2 bench_parallel.cu SPH.cu March.cu -I. -lstdc++ -lm -o bench_parallel
