set(CTEST_BUILD_NAME ${ROOT_ARCHITECTURE}-${CMAKE_BUILD_TYPE})
enable_testing()

CB_ADD_GBENCHMARK(Simple Simple.cpp)
CB_ADD_GBENCHMARK(AlgorithmicComplexity AlgorithmicComplexity.cpp)
CB_ADD_GBENCHMARK(ArrayExpressionTemplates ArrayExpressionTemplates.cpp)
if (CLAD_ENABLE_ENZYME_BACKEND)
  CB_ADD_GBENCHMARK(EnzymeCladComparison EnzymeCladComparison.cpp)
endif(CLAD_ENABLE_ENZYME_BACKEND)

include(FetchContent)
FetchContent_Declare(
  tapenade_kit
  GIT_REPOSITORY https://gitlab.inria.fr/tapenade/tapenade.git
  
  # We use a commit hash from the 'develop' branch because
  # the 'disk offloading' feature is not yet available in the stable release of tapenade.
  GIT_TAG        6f8cb04a4168cac5f6242cdeb65f6816c413f4ee 
  GIT_PROGRESS   TRUE      
)
FetchContent_MakeAvailable(tapenade_kit)

# Build the support library for the stack from the fetched source
add_library(tapenade_support STATIC ${tapenade_kit_SOURCE_DIR}/ADFirstAidKit/adStack.c)
target_include_directories(tapenade_support PUBLIC ${tapenade_kit_SOURCE_DIR}/ADFirstAidKit)

target_compile_definitions(tapenade_support PRIVATE 
      ADSTACK_MAX_SPACES=1        
      ADSTACK_BLOCK_SIZE=4194304   
)

CB_ADD_GBENCHMARK(VectorModeComparison VectorModeComparison.cpp)
CB_ADD_GBENCHMARK(MemoryComplexity_tapenade MemoryComplexity.cpp)
CB_ADD_GBENCHMARK(Multithreading Multithreading.cpp)

target_link_libraries(MemoryComplexity_tapenade PRIVATE tapenade_support)

add_custom_target(benchmark-clad COMMAND ${CMAKE_CTEST_COMMAND} -V
  DEPENDS ${CLAD_BENCHMARK_DEPS} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(benchmark-clad PROPERTIES FOLDER "Clad benchmarks")