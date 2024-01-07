include(ExternalProject)

set(GBENCHMARK_PREFIX "${CMAKE_BINARY_DIR}/googlebenchmark-prefix")
set(GBENCHMARK_LIBRARY_NAME ${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX})

# Remove the coverage flags when compiling external libraries.
string(REPLACE "${GCC_COVERAGE_COMPILE_FLAGS}" "" CMAKE_CXX_FLAGS_NOCOV "${CMAKE_CXX_FLAGS}")
string(REPLACE "${GCC_COVERAGE_COMPILE_FLAGS}" "" CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS_NOCOV "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")
string(REPLACE "${GCC_COVERAGE_LINK_FLAGS}" "" CMAKE_EXE_LINKER_FLAGS_NOCOV "${CMAKE_EXE_LINKER_FLAGS}")
string(REPLACE "${GCC_COVERAGE_LINK_FLAGS}" "" CMAKE_SHARED_LINKER_FLAGS_NOCOV "${CMAKE_SHARED_LINKER_FLAGS}")

# ExternalProject_Add does not like unescaped quotes in CMAKE_ARGS.
string(REPLACE "\"" "\\\"" ESCAPED_CMAKE_CXX_FLAGS_NOCOV "${CMAKE_CXX_FLAGS_NOCOV}")
string(REPLACE "\"" "\\\"" ESCAPED_CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

include(ExternalProject)
#---Find and install google benchmark
ExternalProject_Add(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  EXCLUDE_FROM_ALL 1
  GIT_SHALLOW 1
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
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_NOCOV}
  -DCMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS=${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS_NOCOV}
  -DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS_NOCOV}
  -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS_NOCOV}
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
# Create the directories. Prevents bug https://gitlab.kitware.com/cmake/cmake/issues/15052
file(MAKE_DIRECTORY ${GBENCHMARK_INCLUDE_DIR})

# Libraries
ExternalProject_Get_Property(googlebenchmark binary_dir)
set(_GBENCH_LIBRARY_PATH ${binary_dir}/)

# Register googlebenchmark
add_library(gbenchmark IMPORTED STATIC GLOBAL)
set_target_properties(gbenchmark PROPERTIES
  IMPORTED_LOCATION ${_GBENCH_LIBRARY_PATH}/src/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}
  INTERFACE_INCLUDE_DIRECTORIES "${GBENCHMARK_INCLUDE_DIR}"
  )
target_include_directories(gbenchmark INTERFACE ${GBENCHMARK_INCLUDE_DIR})
add_dependencies(gbenchmark googlebenchmark)
