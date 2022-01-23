include(ExternalProject)

set(GBENCHMARK_PREFIX "${CMAKE_BINARY_DIR}/benchmark/googlebenchmark-prefix")
set(GBENCHMARK_LIBRARY_NAME ${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX})

#---Find and install google benchmark
ExternalProject_Add(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  EXCLUDE_FROM_ALL 1
  GIT_TAG v1.6.0
  UPDATE_COMMAND ""
  # TIMEOUT 10
  # # Force separate output paths for debug and release builds to allow easy
  # # identification of correct lib in subsequent TARGET_LINK_LIBRARIES commands
  # CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
  #            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
  #            -Dgtest_force_shared_crt=ON
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
  -DCMAKE_INSTALL_PREFIX:PATH=${GBENCHMARK_PREFIX}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS=${GBENCHMARK_CMAKE_CXX_FLAGS}
  -DBENCHMARK_ENABLE_TESTING=OFF
  # Disable install step
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${GBENCHMARK_PREFIX}/src/googlebenchmark-build/src/${GBENCHMARK_LIBRARY_NAME}"
  # Wrap download, configure and build steps in a script to log output
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON)

# Specify include dirs for googlebenchmark
ExternalProject_Get_Property(googlebenchmark source_dir)
set(GBENCHMARK_INCLUDE_DIR ${source_dir}/include)

# Libraries
ExternalProject_Get_Property(googlebenchmark binary_dir)
set(_GBENCH_LIBRARY_PATH ${binary_dir}/)

# Register googlebenchmark
add_library(gbenchmark IMPORTED STATIC GLOBAL)
set_property(TARGET gbenchmark PROPERTY IMPORTED_LOCATION ${_GBENCH_LIBRARY_PATH}/src/libbenchmark.a)
add_dependencies(gbenchmark googlebenchmark)
