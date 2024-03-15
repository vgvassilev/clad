cmake_minimum_required(VERSION 3.7.0)

enable_language(CXX)
set(CMAKE_CXX_EXTENSIONS NO)

include(GNUInstallDirs)

# MUST be done before call to clad project
get_cmake_property(_cache_vars CACHE_VARIABLES)
foreach(_cache_var ${_cache_vars})
  get_property(_helpstring CACHE ${_cache_var} PROPERTY HELPSTRING)
  if(_helpstring STREQUAL 
    "No help, variable specified on the command line.")
    set(CMAKE_ARGS "${CMAKE_ARGS} -D${_cache_var}=\"${${_cache_var}}\"")
  endif()
endforeach()

# Generate CMakeArgs.txt file with source, build dir and command line args
write_file("${CMAKE_CURRENT_BINARY_DIR}/CMakeArgs.txt"
  "-S${CMAKE_SOURCE_DIR} -B${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_ARGS}")

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

# If we are not building as a part of LLVM, build clad as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(clad)

  ## Define clad supported version of clang and llvm

  set(CLANG_MIN_SUPPORTED 8.0)
  set(CLANG_MAX_SUPPORTED "18.1.x")
  set(CLANG_VERSION_UPPER_BOUND 18.2.0)
  set(LLVM_MIN_SUPPORTED 8.0)
  set(LLVM_MAX_SUPPORTED "18.1.x")
  set(LLVM_VERSION_UPPER_BOUND 18.2.0)

  if (NOT DEFINED Clang_DIR)
    set(Clang_DIR ${LLVM_DIR})
  endif()
  if (NOT DEFINED LLVM_DIR)
    set(LLVM_DIR ${Clang_DIR})
  endif()

  ## Set Cmake packages search order

  set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
  set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

  ## Find supported LLVM

  if (LLVM_FOUND)
    if (LLVM_PACKAGE_VERSION VERSION_LESS LLVM_MIN_SUPPORTED OR LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL LLVM_VERSION_UPPER_BOUND)
      unset(LLVM_FOUND)
      unset(LLVM_VERSION_MAJOR)
      unset(LLVM_VERSION_MINOR)
      unset(LLVM_VERSION_PATCH)
      unset(LLVM_PACKAGE_VERSION)
    else()
      if (NOT DEFINED LLVM_VERSION AND NOT DEFINED LLVM_DIR)
        set(LLVM_VERSION ${LLVM_PACKAGE_VERSION})
      endif()
    endif()
  endif()

  if (NOT LLVM_FOUND AND DEFINED LLVM_VERSION)
    if (LLVM_VERSION VERSION_GREATER_EQUAL LLVM_VERSION_UPPER_BOUND)
      set(LLVM_VERSION ${LLVM_VERSION_UPPER_BOUND})
    endif()
    if (LLVM_VERSION VERSION_LESS LLVM_MIN_SUPPORTED)
      set(LLVM_VERSION ${LLVM_MIN_SUPPORTED})
    endif()

    if (DEFINED LLVM_DIR)
       set(search_hints HINTS ${LLVM_DIR} "${LLVM_DIR}/lib/cmake/llvm" "${LLVM_DIR}/cmake")
    endif()

    find_package(LLVM ${LLVM_VERSION} REQUIRED CONFIG ${search_hints})
  endif()

  if (NOT LLVM_FOUND AND DEFINED LLVM_DIR)
    find_package(LLVM REQUIRED CONFIG PATHS ${LLVM_DIR} "${LLVM_DIR}/lib/cmake/llvm" "${LLVM_DIR}/cmake" "${LLVM_CONFIG_EXTRA_PATH_HINTS}" NO_DEFAULT_PATH)
  endif()

  if (NOT LLVM_FOUND)
    find_package(LLVM REQUIRED CONFIG)
  endif()

  if (NOT LLVM_FOUND)
    message(FATAL_ERROR "Please set LLVM_DIR pointing to the LLVM build or installation folder")
  endif()

  if (LLVM_PACKAGE_VERSION VERSION_LESS LLVM_MIN_SUPPORTED OR LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL LLVM_VERSION_UPPER_BOUND)
    message(FATAL_ERROR "Found unsupported version: LLVM ${LLVM_PACKAGE_VERSION};\nPlease set LLVM_DIR pointing to the llvm version ${LLVM_MIN_SUPPORTED} to ${LLVM_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  ## Find supported Clang

  if (DEFINED CLANG_VERSION)
    message(FATAL_ERROR "Clang's configuration is not versioned. Please set LLVM_VERSION instead.")
  endif(DEFINED CLANG_VERSION)

  # We have specified -DLLVM_VERSION and we must find matching clang. However,
  # ClangConfig.cmake is not versioned and we will need to work harder to find
  # the correct package.
  if (DEFINED LLVM_VERSION AND NOT DEFINED Clang_DIR)
    set(extra_hints HINTS ${Clang_DIR} "${LLVM_BINARY_DIR}/lib/cmake/clang" "${LLVM_BINARY_DIR}/cmake")
    find_package(Clang REQUIRED CONFIG ${extra_hints} NO_DEFAULT_PATH)
  endif()

  if (NOT Clang_FOUND AND DEFINED Clang_DIR)
    find_package(Clang REQUIRED CONFIG PATHS ${Clang_DIR} "${Clang_DIR}/lib/cmake/clang" "${Clang_DIR}/cmake" "${Clang_CONFIG_EXTRA_PATH_HINTS}" NO_DEFAULT_PATH)
  endif()

  if (NOT Clang_FOUND)
    find_package(Clang REQUIRED CONFIG)
  endif()

  if (NOT Clang_FOUND)
    message(FATAL_ERROR "Please set Clang_DIR pointing to the clang build or installation folder")
  endif()

  set(CLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
  set(CLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
  set(CLANG_VERSION_PATCH ${LLVM_VERSION_PATCH})
  set(CLANG_PACKAGE_VERSION ${LLVM_PACKAGE_VERSION})

  if (CLANG_PACKAGE_VERSION VERSION_LESS CLANG_MIN_SUPPORTED OR CLANG_PACKAGE_VERSION VERSION_GREATER_EQUAL CLANG_VERSION_UPPER_BOUND)
    message(FATAL_ERROR "Found unsupported version: Clang ${CLANG_PACKAGE_VERSION};\nPlease set Clang_DIR pointing to the clang version ${CLANG_MIN_SUPPORTED} to ${CLANG_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: Clang ${CLANG_PACKAGE_VERSION}")
  message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

  if (CLANG_VERSION_MAJOR GREATER_EQUAL 16)
    if (NOT CMAKE_CXX_STANDARD)
      set (CMAKE_CXX_STANDARD 17)
    endif()
    if (CMAKE_CXX_STANDARD LESS 17)
      message(fatal "LLVM/Clad requires C++17 or later")
    endif()
  elseif (CLANG_VERSION_MAJOR GREATER_EQUAL 10)
    if (NOT CMAKE_CXX_STANDARD)
      set (CMAKE_CXX_STANDARD 14)
    endif()
    if (CMAKE_CXX_STANDARD LESS 14)
      message(fatal "LLVM/Clad requires c++14 or later")
    endif()
  endif()

  # When clad is in debug mode the llvm package thinks it is built with -frtti.
  # For consistency we should set it to the correct value.
  set(LLVM_CONFIG_HAS_RTTI NO CACHE BOOL "" FORCE)

  ## Init

  # In case this was a path to a build folder of llvm still try to find AddLLVM
  list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

  # Fix bug in some AddLLVM.cmake implementation (-rpath "" problem)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})

  include(AddLLVM)
  include(HandleLLVMOptions)

  set(CMAKE_INCLUDE_CURRENT_DIR ON)

  # In rare cases we might want to have clang installed in a different place
  # than llvm and the header files should be found first (even though the
  # LLVM_INCLUDE_DIRS) contain clang headers, too.
  include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})

  # If the llvm sources are present add them with higher priority.
  if (LLVM_BUILD_MAIN_SRC_DIR)
    # LLVM_INCLUDE_DIRS contains the include paths to both LLVM's source and
    # build directories. Since we cannot just include ClangConfig.cmake (see
    # fixme above) we have to do a little more work to get the right include
    # paths for clang.
    #
    # FIXME: We only support in-tree builds of clang, that is clang being built
    # in llvm_src/tools/clang.
    include_directories(SYSTEM ${LLVM_BUILD_MAIN_SRC_DIR}/tools/clang/include/)

    if (NOT LLVM_BUILD_BINARY_DIR)
      message(FATAL "LLVM_BUILD_* values should be available for the build tree")
    endif()

    include_directories(SYSTEM ${LLVM_BUILD_BINARY_DIR}/tools/clang/include/)
  endif()

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

  set( CLAD_BUILT_STANDALONE 1 )
endif()

## Code Coverage Configuration
add_library(coverage_config INTERFACE)
option(CLAD_CODE_COVERAGE "Enable coverage reporting" OFF)
if(CLAD_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
  if(NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message(FATAL_ERROR "CodeCov enabled on non-debug build!")
  endif()
  set(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
  set(GCC_COVERAGE_LINK_FLAGS    "--coverage")
  set(CMAKE_CXX_FLAGS            "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS     "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHAREDLINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
endif()

##
include(Compatibility.cmake)

set(C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories clad will search for headers.")

set(CLAD_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLAD_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLVM. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

if (APPLE)
  set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-undefined -Wl,dynamic_lookup")
endif ()

# FIXME: Use merge this with the content from the LLVMConfig and ClangConfig.
if (NOT CLAD_BUILT_STANDALONE)
include_directories(BEFORE SYSTEM
  ${CMAKE_CURRENT_BINARY_DIR}/../clang/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../clang/include
  )
endif()

include_directories(BEFORE SYSTEM
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )

install(DIRECTORY tools/
  DESTINATION include/clad/tools
  FILES_MATCHING
  PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "*.inc"
  )

add_definitions( -D_GNU_SOURCE
  -DCLAD_SRCDIR_INCL="${CLAD_SOURCE_DIR}/include"
  -DCLAD_INSTDIR_INCL="${CLAD_BINARY_DIR}/include" )

option(CLAD_BUILD_STATIC_ONLY "Does not build shared libraries. Useful when we have LLVM_ENABLE_PLUGINS=OFF (eg. Win or CYGWIN)" OFF)
if (NOT CLAD_BUILD_STATIC_ONLY AND NOT LLVM_ENABLE_PLUGINS)
  message(FATAL_ERROR "LLVM_ENABLE_PLUGINS is set to OFF. Please build clad with -DCLAD_BUILD_STATIC_ONLY=ON.")
endif()

# Add clad deps if we build together with clang.
if (TARGET intrinsics_gen)
  list(APPEND LLVM_COMMON_DEPENDS intrinsics_gen)
endif()
if (TARGET clang-headers)
  list(APPEND LLVM_COMMON_DEPENDS clang-headers)
endif()

add_subdirectory(lib)
add_subdirectory(tools)

option(CLAD_INCLUDE_DOCS "Generate build targets for the clad docs.")
# Should we instead name the options as CLAD_INTERNAL_DOCS and CLAD_EXTERNAL_DOCS?
option(CLAD_ENABLE_DOXYGEN "Use doxygen to generate clad interal API documentation.")
option(CLAD_ENABLE_SPHINX "Use sphinx to generage clad user documentation")
if (CLAD_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()


if (NOT CLAD_BUILD_STATIC_ONLY)
  include(AddClad)

  if (CLAD_ENABLE_BENCHMARKS)
    include(GoogleBenchmark)
  endif(CLAD_ENABLE_BENCHMARKS)

  # Change the default compiler to the clang which we run clad upon. Our unittests
  # need to use a supported by clad compiler. Note that's a huge hack and it is
  # not guaranteed to work with cmake.
  set(stored_cxx_compiler ${CMAKE_CXX_COMPILER})
  set(stored_cxx_flags ${CMAKE_CXX_FLAGS})

  set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
  # Filter some unsupported flags by clang.
  string(REPLACE "-fno-lifetime-dse" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Wno-class-memaccess" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  add_subdirectory(unittests)
  add_subdirectory(test)
  add_subdirectory(demos/ErrorEstimation/CustomModel)
  add_subdirectory(demos/ErrorEstimation/PrintModel)

  # Add benchmarking infrastructure.
  if (CLAD_ENABLE_BENCHMARKS)
    add_subdirectory(benchmark)
  endif(CLAD_ENABLE_BENCHMARKS)

  # Restore the default compiler.
  set(CMAKE_CXX_COMPILER ${stored_cxx_compiler})
  set(CMAKE_CXX_FLAGS ${stored_cxx_flags})
endif()

# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
if( CLAD_BUILT_STANDALONE AND MSVC_VERSION EQUAL 1600 )
  set(CLAD_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/clad.sln")
  if( EXISTS "${CLAD_SLN_FILENAME}" )
    file(APPEND "${CLAD_SLN_FILENAME}" "\n# This should be regenerated!\n")
  endif()
endif()
