cmake_minimum_required(VERSION 3.16.3)

project(clad_example)

add_executable ( clad_example main.cpp )
find_package(Kokkos REQUIRED)
set (CMAKE_CXX_STANDARD 17)
target_include_directories( clad_example PRIVATE ${CLAD_INCLUDE_PATH})
target_link_libraries(clad_example PRIVATE Kokkos::kokkos)
