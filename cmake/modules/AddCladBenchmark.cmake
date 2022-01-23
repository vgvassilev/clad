# Find the current branch.
execute_process(WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMAND git rev-parse --abbrev-ref HEAD
                OUTPUT_VARIABLE CURRENT_REPO_BRANCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "/" "" CURRENT_REPO_BRANCH ${CURRENT_REPO_BRANCH})

# Ask cmake to reconfigure each time we change the branch so that it can change
# the value of CURRENT_REPO_BRANCH.
set_property(DIRECTORY APPEND PROPERTY
             CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/.git/HEAD")


# Change the default compiler to the clang which we run clad upon.
set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)

#----------------------------------------------------------------------------
# function CB_ADD_GBENCHMARK(<benchmark> source1 source2... LIBRARIES libs)
#----------------------------------------------------------------------------
function(CB_ADD_GBENCHMARK benchmark)
  cmake_parse_arguments(ARG "" "LABEL" "DEPENDS;LIBRARIES" ${ARGN})
  set(source_files ${ARG_UNPARSED_ARGUMENTS})

  add_executable(${benchmark} ${source_files})

  # Add the clad plugin
  target_compile_options(${benchmark} PUBLIC -fplugin=$<TARGET_FILE:clad>)

  # Debugging. Emitting the derivatives' source code.
  #target_compile_options(${benchmark} PUBLIC "SHELL:-Xclang -plugin-arg-clad"
  #  "SHELL: -Xclang -fdump-derived-fn")

  # Debugging. Emit llvm IR.
  #target_compile_options(${benchmark} PUBLIC -S -emit-llvm)

  # Debugging. Optimization misses.
  #target_compile_options(${benchmark} PUBLIC "SHELL:-Xclang -Rpass-missed=.*inline.*")

  # Optimize the produced code.
  target_compile_options(${benchmark} PUBLIC -O3)

  # Turn off numerical diff fallback.
  target_compile_definitions(${benchmark} PUBLIC CLAD_NO_NUM_DIFF)

  # Clad requires us to link against these libraries.
  target_link_libraries(${benchmark} PUBLIC stdc++ pthread m)

  target_include_directories(${benchmark} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${GBENCHMARK_INCLUDE_DIR})
  set_property(TARGET ${benchmark} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  target_link_libraries(${benchmark} PUBLIC ${ARG_LIBRARIES} gbenchmark)
  if (NOT APPLE)
    target_link_libraries(${benchmark} PUBLIC rt)
  endif()

  add_dependencies(${benchmark} clad)
  # If clad.so changes we don't need to relink but to rebuild the source files.
  # $<TARGET_FILE:clad> does not work for OBJECT_DEPENDS.
  set (CLAD_SO_PATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/clad${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set_source_files_properties(${source_files} PROPERTY OBJECT_DEPENDS ${CLAD_SO_PATH})

  set (TIMEOUT_VALUE 1200)
  set (LABEL "short")
  if (ARG_LABEL AND "${ARG_LABEL}" STREQUAL "long")
    set (TIMEOUT_VALUE 2400)
    set (LABEL "long")
  endif()

  # Add dependencies to benchmark
  if(ARG_DEPENDS)
    add_dependencies(${benchmark} ${ARG_DEPENDS})
  endif()

  # Add benchmark as a CTest
  add_test(NAME clad-${benchmark}
    COMMAND ${benchmark} --benchmark_out_format=json
    --benchmark_out=clad-gbenchmark-${benchmark}-${CURRENT_REPO_BRANCH}.json
    --benchmark_color=false)
  set_tests_properties(clad-${benchmark} PROPERTIES
                       TIMEOUT "${TIMEOUT_VALUE}"
                       LABELS "benchmark;${LABEL}"
                       RUN_SERIAL TRUE
                       DEPENDS ${benchmark})
endfunction(CB_ADD_GBENCHMARK)
