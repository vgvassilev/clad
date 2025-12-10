set(_gtest_byproduct_binary_dir
  ${CLAD_BINARY_DIR}/unittests/googletest-prefix/src/googletest-build)
set(_gtest_byproducts
  ${_gtest_byproduct_binary_dir}/lib/libgtest.a
  ${_gtest_byproduct_binary_dir}/lib/libgtest_main.a
  ${_gtest_byproduct_binary_dir}/lib/libgmock.a
  ${_gtest_byproduct_binary_dir}/lib/libgmock_main.a
  )

if(MSVC)
  set(EXTRA_GTEST_OPTS
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=${_gtest_byproduct_binary_dir}/lib/
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO:PATH=${_gtest_byproduct_binary_dir}/lib/
    -Dgtest_force_shared_crt=ON)
elseif(APPLE)
  set(EXTRA_GTEST_OPTS -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
                       -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})
endif()

set(EXTRA_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} ${EXTRA_GTEST_OPTS}")
clad_externalproject_add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release -- -j6
  # Disable install step
  INSTALL_COMMAND cmake -E echo "Skipping install step."
  BUILD_BYPRODUCTS ${_gtest_byproducts}
  # Wrap download, configure and build steps in a script to log output
  EXTRA_CMAKE_ARGS ${EXTRA_CMAKE_ARGS}
  )

# Specify include dirs for gtest and gmock
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIR ${source_dir}/googletest/include)
set(GMOCK_INCLUDE_DIR ${source_dir}/googlemock/include)
# Create the directories. Prevents bug https://gitlab.kitware.com/cmake/cmake/issues/15052
file(MAKE_DIRECTORY ${GTEST_INCLUDE_DIR} ${GMOCK_INCLUDE_DIR})

# Libraries
ExternalProject_Get_Property(googletest binary_dir)
set(_G_LIBRARY_PATH ${binary_dir}/lib/)

# Use gmock_main instead of gtest_main because it initializes gtest as well.
# Note: The libraries are listed in reverse order of their dependancies.
foreach(lib gtest gtest_main gmock gmock_main)
  add_library(${lib} IMPORTED STATIC GLOBAL)
  set_target_properties(${lib} PROPERTIES
    IMPORTED_LOCATION "${_G_LIBRARY_PATH}${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
    )
  add_dependencies(${lib} googletest)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND
      ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 9)
    target_compile_options(${lib} INTERFACE -Wno-deprecated-copy)
  endif()
endforeach()
target_include_directories(gtest INTERFACE ${GTEST_INCLUDE_DIR})
target_include_directories(gmock INTERFACE ${GMOCK_INCLUDE_DIR})

set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET gmock PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gmock${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET gmock_main PROPERTY IMPORTED_LOCATION ${_G_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}gmock_main${CMAKE_STATIC_LIBRARY_SUFFIX})
