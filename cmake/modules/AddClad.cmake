if (CLAD_ENABLE_BENCHMARKS)
  # Find the current branch.
  execute_process(WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                  COMMAND git rev-parse HEAD
                  OUTPUT_VARIABLE CURRENT_REPO_COMMIT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "/" "" CURRENT_REPO_COMMIT ${CURRENT_REPO_COMMIT})

# Ask cmake to reconfigure each time we change the branch so that it can change
# the value of CURRENT_REPO_COMMIT.
set_property(DIRECTORY APPEND PROPERTY
             CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/.git/HEAD")

endif(CLAD_ENABLE_BENCHMARKS)

#-------------------------------------------------------------------------------
# function ENABLE_CLAD_FOR_TARGET(<executable>
#   DEPENDS dependencies...
#     A list of targets that the executable depends on.
#   LIBRARIES libraries...
#     A list of libraries to be linked in. Defaults to stdc++ pthread m.
# )
#-------------------------------------------------------------------------------
function(ENABLE_CLAD_FOR_TARGET executable)
  if (NOT TARGET ${executable})
    message(FATAL_ERROR "'${executable}' is not a valid target.")
  endif()

  # Add the clad plugin
  target_compile_options(${executable} PUBLIC -fplugin=$<TARGET_FILE:clad>)

  # Debugging. Emitting the derivatives' source code.
  #target_compile_options(${executable} PUBLIC "SHELL:-Xclang -plugin-arg-clad"
  #  "SHELL: -Xclang -fdump-derived-fn")

  # Debugging. Emit llvm IR.
  #target_compile_options(${executable} PUBLIC -S -emit-llvm)

  # Debugging. Optimization misses.
  #target_compile_options(${executable} PUBLIC "SHELL:-Xclang -Rpass-missed=.*inline.*")

  # Clad requires us to link against these libraries.
  target_link_libraries(${executable} PUBLIC stdc++ pthread m)

  target_include_directories(${executable} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TARGET ${executable} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(${executable} clad)
  # If clad.so changes we don't need to relink but to rebuild the source files.
  # $<TARGET_FILE:clad> does not work for OBJECT_DEPENDS.
  set (CLAD_SO_PATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/clad${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set_source_files_properties(${source_files} PROPERTY OBJECT_DEPENDS ${CLAD_SO_PATH})

  # Add dependencies to executable
  if(ARG_DEPENDS)
    add_dependencies(${executable} ${ARG_DEPENDS})
  endif(ARG_DEPENDS)

endfunction(ENABLE_CLAD_FOR_TARGET)

#-------------------------------------------------------------------------------
# function ADD_CLAD_LIBRARY(<library> sources...
#   DEPENDS dependencies...
#     A list of targets that the library depends on.
#   LIBRARIES libraries...
#     A list of libraries to be linked in. Defaults to stdc++ pthread m.
# )
#-------------------------------------------------------------------------------
function(ADD_CLAD_LIBRARY library)
  cmake_parse_arguments(ARG "" "DEPENDS;LIBRARIES" "" ${ARGN})

  set(source_files ${ARG_UNPARSED_ARGUMENTS})

  add_library (${library} ${source_files})
  ENABLE_CLAD_FOR_TARGET(${library} ${ARGN})

endfunction(ADD_CLAD_LIBRARY)


#-------------------------------------------------------------------------------
# function ADD_CLAD_EXECUTABLE(<executable> sources...
#   DEPENDS dependencies...
#     A list of targets that the executable depends on.
#   LIBRARIES libraries...
#     A list of libraries to be linked in. Defaults to stdc++ pthread m.
# )
#-------------------------------------------------------------------------------
function(ADD_CLAD_EXECUTABLE executable)
  cmake_parse_arguments(ARG "" "DEPENDS;LIBRARIES" "" ${ARGN})

  set(source_files ${ARG_UNPARSED_ARGUMENTS})

  add_executable(${executable} ${source_files})

  ENABLE_CLAD_FOR_TARGET(${executable} ${ARGN})

endfunction(ADD_CLAD_EXECUTABLE)

#-------------------------------------------------------------------------------
# function CB_ADD_GBENCHMARK(<benchmark> sources
#   LABEL <short|long>
#     A label that classifies how much time a benchmark is expected to take.
#     Short benchmarks time out at 1200 seconds, long at 2400.
#   DEPENDS dependencies...
#     A list of targets that the executable depends on.
#   LIBRARIES libraries...
#     A list of libraries to be linked in. Defaults to stdc++ pthread m.
# )
#-------------------------------------------------------------------------------
function(CB_ADD_GBENCHMARK benchmark)
  cmake_parse_arguments(ARG "" "LABEL" "" ${ARGN})
  ADD_CLAD_EXECUTABLE(${benchmark} ${ARG_UNPARSED_ARGUMENTS})

  # Optimize the produced code.
  target_compile_options(${benchmark} PUBLIC -O3)

  # Turn off numerical diff fallback.
  target_compile_definitions(${benchmark} PUBLIC CLAD_NO_NUM_DIFF)

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

  # Add benchmark as a CTest
  add_test(NAME clad-${benchmark}
    COMMAND ${benchmark} --benchmark_out_format=json
    --benchmark_out=clad-gbenchmark-${benchmark}-${CURRENT_REPO_COMMIT}.json
    --benchmark_color=false)
  set_tests_properties(clad-${benchmark} PROPERTIES
                       TIMEOUT "${TIMEOUT_VALUE}"
                       LABELS "benchmark;${LABEL}"
                       RUN_SERIAL TRUE
                       DEPENDS ${benchmark})
endfunction(CB_ADD_GBENCHMARK)
