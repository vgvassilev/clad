add_custom_target(CladUnitTests)
set_target_properties(CladUnitTests PROPERTIES FOLDER "Clad tests")

# LLVM builds (not installed llvm) provides gtest.
if (NOT TARGET gtest)
  include(CladGoogleTest)
endif()

# add_clad_unittest(test_dirname file1.cpp file2.cpp)
#
# Will compile the list of files together and link against clad.
# Produces a binary named 'basename(test_dirname)'.
function(add_clad_unittest test_dirname)

  add_unittest(CladUnitTests ${test_dirname} ${ARGN})

  # Remove the llvm_gtest_* coming from add_unittest.
  get_target_property(GTEST_LINKED_LIBS ${test_dirname} LINK_LIBRARIES)
  list(REMOVE_ITEM GTEST_LINKED_LIBS llvm_gtest_main llvm_gtest)
  set_property(TARGET ${test_dirname} PROPERTY LINK_LIBRARIES ${GTEST_LINKED_LIBS})

  enable_clad_for_target(${test_dirname})

  # Add gtest dependencies.
  set(gtest_libs gtest gtest_main)
  # Clang prior than clang13 (I think) merges both gmock into gtest.
  if (TARGET gmock)
    list(APPEND gtest_libs gmock gmock_main)
  endif()

  target_link_libraries(${test_dirname} PUBLIC ${gtest_libs})

endfunction()

add_subdirectory(Basic)

find_package(Kokkos)
if (Kokkos_FOUND)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  add_subdirectory(Kokkos)
endif(Kokkos_FOUND)

add_subdirectory(Misc)
