# to compile, go to the folder containing this file and run:
# cmake -S . -B build
# cmake --build build
# this will create a build folder with executables
# to run type :
# ./build/merge


#thnx AI

cmake_minimum_required(VERSION 3.14)

project(merge CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Sources ─────────────────────────────────────────────

set(SOURCES
    src/main.cpp
    src/rayTrace.cpp
    src/March.cpp
    src/SPH.cpp
)

add_executable(merge ${SOURCES})

# Include headers from src/

target_include_directories(merge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ── Compiler flags ──────────────────────────────────────

target_compile_options(merge PRIVATE
    -Wall
    -Wextra
    -O2
)

# Math library (Linux mostly; safe to keep)

target_link_libraries(merge PRIVATE m)

# ── Build type ──────────────────────────────────────────

if(NOT CMAKE_BUILD_TYPE)

    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

endif()

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")

message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")