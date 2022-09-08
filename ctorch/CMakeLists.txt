cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

set(CMAKE_CXX_FLAGS "-O3 -ffast-math")

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)

# add_executable(copp copp.cpp)
# target_link_libraries(copp "${TORCH_LIBRARIES}")
# set_property(TARGET copp PROPERTY CXX_STANDARD 14)
