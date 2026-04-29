<img width="1308" height="418" alt="Screenshot from 2026-04-23 22-47-59" src="https://github.com/user-attachments/assets/2f56fdf3-507d-43a0-bf5f-e3c50702a86d" />


How to Run the Serial and Parallel Versions

Remaining of file was created by AI (mostly)
==========================================

Project Structure
-----------------
- The serial CPU implementation is in the `merged/` folder.
- The parallel GPU implementation is in the `merged_parallel/` folder.

Serial Version
--------------
The serial version runs on the CPU and includes:
- Serial brute-force ray tracing
- Serial DDA-accelerated ray tracing
- Benchmark timing for ray tracing + rendering

Steps:

1. Go to the serial project folder:

   cd merged

2. Build the project:

   cmake -S . -B build
   cmake --build build

3. Run the executable:

   ./build/merge

4. The program will print benchmark information for each frame, including:

   - physics benchmarking
   - mesh benchmarking
   - DDA render benchmarking
   - brute-force render benchmarking

Example output sections:

   DDA render benchmarking:
       construct triangles
       AABB voxel binning
       CSR flatten
       ray trace + shading
       aa + pack + write
       rays traced

   Brute-force render benchmarking:
       construct triangles
       ray trace + shading
       aa + write
       rays traced

5. To stop the serial program, press:

   Ctrl + C


Parallel Version
----------------
The parallel version runs on the GPU using CUDA and includes:
- GPU SPH physics
- GPU Marching Cubes
- GPU DDA ray tracing
- Benchmark timing for each pipeline stage

Steps:

1. Go to the parallel project folder:

   cd merged_parallel

2. Compile with nvcc:

   nvcc -ccbin /usr/bin/gcc-14 -std=c++17 -O2 main.cu SPH.cu March.cu rayTrace.cu -I. -lstdc++ -lm -o physics_sim

3. Run the parallel benchmark without writing image frames:

   ./physics_sim --frames=10 --no-output

4. Run the parallel benchmark while writing image frames:

   ./physics_sim --frames=10

5. Optional command-line flags:

   --frames=N
       Number of frames to generate or benchmark.

   --substeps=N
       Number of physics substeps per rendered frame.

   --boxdim=N
       DDA voxel grid dimension. For example:
       ./physics_sim --frames=10 --boxdim=32 --no-output

   --no-output
       Disables writing .ppm image files. Useful for timing without file I/O.

   --bruteforce
       Runs brute-force ray tracing mode instead of DDA mode.

   --with-sphere
       Adds the test sphere to the scene.

   --csv=PATH
       Saves benchmark CSV output to a custom path.

Example:

   ./physics_sim --frames=10 --boxdim=32 --no-output --csv=benchmark.csv

6. The program writes benchmark results to:

   benchmark.csv
   benchmark.txt

   the txt file represents a summary for the information gathered. All information can ve retrieved via the csv file.

unless a custom CSV path is given.


Nsight Compute Profiling
------------------------
To profile the GPU ray-tracing kernel:

1. Go to the parallel folder:

   cd merged_parallel

2. Run Nsight Compute on the ray kernel:

   ncu --kernel-name computeRayColors ./physics_sim --frames=10 --no-output

3. To collect cache hit rates:

   ncu --kernel-name computeRayColors \
       --metrics l1tex__t_sector_hit_rate,lts__t_sector_hit_rate \
       ./physics_sim --frames=10 --no-output 2>&1 | tee ray_cache.txt

4. To collect Speed of Light / roofline-related information:

   ncu --kernel-name computeRayColors \
       --section SpeedOfLight_RooflineChart \
       ./physics_sim --frames=10 --no-output 2>&1 | tee ray_roofline.txt

Notes:
- Nsight profiling adds significant overhead, so do not use Nsight timing as normal runtime.
- Use regular benchmark runs for runtime numbers.
- Use Nsight only for profiling metrics such as throughput and cache hit rates.


Converting Frames to Video
--------------------------
If image frames are written to the `frames/` directory (which they should be), they can be converted to a video with ffmpeg:

   ffmpeg -framerate 60 -i frames/image%d.ppm -c:v libx264 -vf "format=yuv420p" output.mp4


Important Benchmarking Notes
----------------------------
- Serial brute-force vs serial DDA measures the algorithmic speedup from DDA.
- Serial DDA vs GPU DDA measures the GPU parallelization speedup.
- Serial brute-force vs GPU DDA measures combined algorithmic + parallel speedup.
- For fair ray-tracing comparisons, compare implementations using the same scene, frame count, resolution, and DDA box dimension.
