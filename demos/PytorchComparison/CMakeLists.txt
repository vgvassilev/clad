cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch-basic-nn)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torch-basic-nn torch-basic-nn.cpp)
target_link_libraries(torch-basic-nn "${TORCH_LIBRARIES}")
set_property(TARGET torch-basic-nn PROPERTY CXX_STANDARD 17)

