
# require CMake 3.10 or greater
cmake_minimum_required(VERSION 3.10)

# declare my-recognition project
project(my-recognition)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

# find VPI package (optional)
# each VPI major version ships its own cmake package that only accepts a
# request with a matching major version, so probe them newest-first
# (VPI 4 = JetPack 7, VPI 3 = JetPack 6.1+, VPI 2 = JetPack 5/6)
foreach(VPI_MAJOR 4 3 2)
	if(NOT VPI_FOUND)
		find_package(VPI ${VPI_MAJOR}.0 QUIET)
	endif()
endforeach()

# CUDA is required
find_package(CUDA)

# add directory for libnvbuf-utils to program
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# compile the my-recognition program
cuda_add_executable(my-recognition my-recognition.cpp)

# link my-recognition to jetson-inference library
target_link_libraries(my-recognition jetson-inference)
