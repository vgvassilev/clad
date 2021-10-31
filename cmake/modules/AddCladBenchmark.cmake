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

  # Debugging
  #target_compile_options(${benchmark} PUBLIC "SHELL:-Xclang -plugin-arg-clad"
  #  "SHELL: -Xclang -fdump-derived-fn")

  # Optimize the produced code
  target_compile_options(${benchmark} PUBLIC -O2)
  add_dependencies(${benchmark} clad)
  # Clad requires us to link against these libraries.
  target_link_libraries(${benchmark} PUBLIC stdc++ pthread m)

  target_include_directories(${benchmark} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${GBENCHMARK_INCLUDE_DIR})
  set_property(TARGET ${benchmark} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  target_link_libraries(${benchmark} PUBLIC ${ARG_LIBRARIES} gbenchmark)
  if (NOT APPLE)
    target_link_libraries(${benchmark} PUBLIC rt)
  endif()
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
           COMMAND ${benchmark} --benchmark_out_format=csv --benchmark_out=clad-gbenchmark-${benchmark}.csv --benchmark_color=false)
  set_tests_properties(clad-${benchmark} PROPERTIES
                       TIMEOUT "${TIMEOUT_VALUE}" LABELS "${LABEL}" RUN_SERIAL TRUE)
endfunction(CB_ADD_GBENCHMARK)
