set(CTEST_BUILD_NAME ${ROOT_ARCHITECTURE}-${CMAKE_BUILD_TYPE})
enable_testing()

CB_ADD_GBENCHMARK(Simple Simple.cpp)
CB_ADD_GBENCHMARK(AlgorithmicComplexity AlgorithmicComplexity.cpp)
CB_ADD_GBENCHMARK(ArrayExpressionTemplates ArrayExpressionTemplates.cpp)
if (CLAD_ENABLE_ENZYME_BACKEND)
  CB_ADD_GBENCHMARK(EnzymeCladComparison EnzymeCladComparison.cpp)
endif(CLAD_ENABLE_ENZYME_BACKEND)
CB_ADD_GBENCHMARK(VectorModeComparison VectorModeComparison.cpp)
CB_ADD_GBENCHMARK(MemoryComplexity MemoryComplexity.cpp)
CB_ADD_GBENCHMARK(Hessians Hessians.cpp)
CB_ADD_GBENCHMARK(GPT2Training GPT2Training.cpp)

if(APPLE)
  # On macOS, we want to explicitly use the high-performance Accelerate framework for BLAS.
  # We also need to give CMake a hint to find the Homebrew OpenMP library.
  list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
  find_package(OpenMP REQUIRED)
  # Link explicitly against Accelerate and the found OpenMP library.
  target_link_libraries(GPT2Training PRIVATE "-framework Accelerate" OpenMP::OpenMP_CXX)
else()
  # On all other platforms (e.g., Linux), find the best available BLAS and OpenMP.
  find_package(BLAS REQUIRED)
  find_package(OpenMP REQUIRED)
  target_link_libraries(GPT2Training PRIVATE ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
endif()
target_compile_options(GPT2Training PRIVATE -O3 -ffast-math)
target_compile_definitions(GPT2Training PRIVATE OMP)
if (BLAS_FOUND)
    target_compile_definitions(GPT2Training PRIVATE HAVE_CBLAS)
endif()

set (CLAD_BENCHMARK_DEPS clad)
get_property(_benchmark_names DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)

foreach (name ${_benchmark_names})
  get_test_property(${name} LABELS _labels)
  if (_labels MATCHES ".*benchmark.*")
    get_test_property(${name} DEPENDS _deps)
    list(APPEND CLAD_BENCHMARK_DEPS ${_deps})
  endif()
endforeach()

add_custom_target(benchmark-clad COMMAND ${CMAKE_CTEST_COMMAND} -V
  DEPENDS ${CLAD_BENCHMARK_DEPS} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(benchmark-clad PROPERTIES FOLDER "Clad benchmarks")
