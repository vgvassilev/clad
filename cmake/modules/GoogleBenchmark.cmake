set(GBENCHMARK_PREFIX "${CMAKE_BINARY_DIR}/googlebenchmark-prefix")
set(GBENCHMARK_LIBRARY_NAME ${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX})

#---Find and install google benchmark
set(EXTRA_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${GBENCHMARK_PREFIX} -DBENCHMARK_ENABLE_TESTING=OFF")
clad_externalProject_add(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
  UPDATE_COMMAND ""
  EXTRA_CMAKE_ARGS "${EXTRA_CMAKE_ARGS}"
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release -- -j6
  # Disable install step
  INSTALL_COMMAND cmake -E echo "Skipping install step."
  BUILD_BYPRODUCTS "${GBENCHMARK_PREFIX}/src/googlebenchmark-build/src/${GBENCHMARK_LIBRARY_NAME}"
)

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
