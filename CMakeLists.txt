cmake_minimum_required(VERSION 3.10)
project(ffvideo_python)

# Force compilation for ARM64 (Apple Silicon)
set(CMAKE_OSX_ARCHITECTURES arm64)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages via Homebrew
find_package(dlib REQUIRED)
find_package(Boost REQUIRED)
find_package(JPEG REQUIRED)

# Add the shared library
add_library(ffvideo_python SHARED
    ffvideo_player_src/ffvideo_python.cpp
)

# Link libraries
target_link_libraries(ffvideo_python PRIVATE
    dlib::dlib
    Boost::boost
    JPEG::JPEG
)

# Include directories for our header
target_include_directories(ffvideo_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ffvideo_player_src
)
