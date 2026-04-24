cmake_minimum_required(VERSION 3.18)

# cuda is enabled conditionally below
project(particle_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# serial (cpu-only) target
add_executable(particle_sim
    src/serial/SPH.cpp
    src/serial/main.cpp
    src/serial/March.cpp
)
target_include_directories(particle_sim PRIVATE src/serial)
target_link_libraries(particle_sim PRIVATE sfml-graphics sfml-window sfml-system)
target_compile_options(particle_sim PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -fsanitize=address>
)

# serial benchmark (headless, no sfml)
add_executable(bench_serial
    src/serial/SPH.cpp
    src/serial/bench.cpp
    src/serial/March.cpp
)
target_include_directories(bench_serial PRIVATE src/serial)
target_compile_options(bench_serial PRIVATE $<$<CONFIG:Release>:-O2>)

# parallel (cuda) target
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    add_executable(particle_sim_parallel
        src/parallel/SPH.cu
        src/parallel/main.cpp
        src/parallel/March.cpp
    )
    target_include_directories(particle_sim_parallel PRIVATE src/parallel)
    target_link_libraries(particle_sim_parallel PRIVATE sfml-graphics sfml-window sfml-system)
    target_compile_options(particle_sim_parallel PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O2>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g -fsanitize=address>
    )
    message(STATUS "CUDA found — particle_sim_parallel target enabled")
else()
    message(STATUS "CUDA not found — skipping particle_sim_parallel target")
endif()
