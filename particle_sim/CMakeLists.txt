cmake_minimum_required(VERSION 3.14)
project(particle_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

add_executable(particle_sim
    src/SPH.cpp
    src/main.cpp
    src/March.cpp
)

target_include_directories(particle_sim PRIVATE src)
target_link_libraries(particle_sim PRIVATE sfml-graphics sfml-window sfml-system)

# optimize
target_compile_options(particle_sim PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -fsanitize=address>
)
