# If we are not building as a part of LLVM, build clad as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(clad)
  cmake_minimum_required(VERSION 3.7.0)

  set(CLANG_MIN_SUPPORTED 5.0)
  set(CLANG_MAX_SUPPORTED 7.1)
  set(LLVM_MIN_SUPPORTED 5.0)
  set(LLVM_MAX_SUPPORTED 7.1)

  if (NOT DEFINED Clang_DIR)
    set(Clang_DIR ${LLVM_DIR})
  endif()
  if (NOT DEFINED LLVM_DIR)
    set(LLVM_DIR ${Clang_DIR})
  endif()

  ## Find supported Clang

  set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
  set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

  if (DEFINED CLANG_VERSION)
    if (CLANG_VERSION VERSION_GREATER CLANG_MAX_SUPPORTED)
      set(CLANG_VERSION ${CLANG_MAX_SUPPORTED})
    endif()
    if (CLANG_VERSION VERSION_LESS CLANG_MIN_SUPPORTED)
      set(CLANG_VERSION ${CLANG_MIN_SUPPORTED})
    endif()

    find_package(Clang ${CLANG_VERSION} REQUIRED CONFIG)
  endif()

  if (NOT Clang_FOUND AND DEFINED Clang_DIR)
    find_package(Clang REQUIRED CONFIG PATHS ${Clang_DIR} "${Clang_DIR}/lib/cmake/clang" "${Clang_DIR}/cmake" NO_DEFAULT_PATH)
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

  if (CLANG_PACKAGE_VERSION VERSION_LESS CLANG_MIN_SUPPORTED OR CLANG_PACKAGE_VERSION VERSION_GREATER CLANG_MAX_SUPPORTED)
    message(FATAL_ERROR "Found unsupported version: Clang ${CLANG_PACKAGE_VERSION};\nPlease set Clang_DIR pointing to the clang version ${CLANG_MIN_SUPPORTED} to ${CLANG_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: Clang ${CLANG_PACKAGE_VERSION}")
  message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

  ## Find supported LLVM

  if (LLVM_FOUND)
    if (LLVM_PACKAGE_VERSION VERSION_LESS LLVM_MIN_SUPPORTED OR LLVM_PACKAGE_VERSION VERSION_GREATER LLVM_MAX_SUPPORTED)
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
    if (LLVM_VERSION VERSION_GREATER LLVM_MAX_SUPPORTED)
      set(LLVM_VERSION ${LLVM_MAX_SUPPORTED})
    endif()
    if (LLVM_VERSION VERSION_LESS LLVM_MIN_SUPPORTED)
      set(LLVM_VERSION ${LLVM_MIN_SUPPORTED})
    endif()

    find_package(LLVM ${LLVM_VERSION} REQUIRED CONFIG)
  endif()

  if (NOT LLVM_FOUND AND DEFINED LLVM_DIR)
    find_package(LLVM REQUIRED CONFIG PATHS ${LLVM_DIR} "${LLVM_DIR}/lib/cmake/llvm" "${LLVM_DIR}/cmake" NO_DEFAULT_PATH)
  endif()

  if (NOT LLVM_FOUND)
    find_package(LLVM REQUIRED CONFIG)
  endif()

  if (NOT LLVM_FOUND)
    message(FATAL_ERROR "Please set LLVM_DIR pointing to the LLVM build or installation folder")
  endif()

  if (LLVM_PACKAGE_VERSION VERSION_LESS LLVM_MIN_SUPPORTED OR LLVM_PACKAGE_VERSION VERSION_GREATER LLVM_MAX_SUPPORTED)
    message(FATAL_ERROR "Found unsupported version: LLVM ${LLVM_PACKAGE_VERSION};\nPlease set LLVM_DIR pointing to the llvm version ${LLVM_MIN_SUPPORTED} to ${LLVM_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

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
  include_directories(${CLANG_INCLUDE_DIRS})
  include_directories(${LLVM_INCLUDE_DIRS})
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
    include_directories(${LLVM_BUILD_MAIN_SRC_DIR}/tools/clang/include/)

    if (NOT LLVM_BUILD_BINARY_DIR)
      message(FATAL "LLVM_BUILD_* values should be available for the build tree")
    endif()

    include_directories(${LLVM_BUILD_BINARY_DIR}/tools/clang/include/)
  endif()

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

  set( CLAD_BUILT_STANDALONE 1 )
endif()

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

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -pedantic -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings")
endif ()

if (APPLE)
  set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif ()

# FIXME: Use merge this with the content from the LLVMConfig and ClangConfig.
if (NOT CLAD_BUILT_STANDALONE)
include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/../clang/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../clang/include
  )
endif()

include_directories(BEFORE
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

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "*.inc"
  )

add_definitions( -D_GNU_SOURCE
  -DCLAD_SRCDIR_INCL="${CLAD_SOURCE_DIR}/include"
  -DCLAD_INSTDIR_INCL="${CLAD_BINARY_DIR}/include" )

include(VersionFromVCS)
# SVN_REVISION and GIT_COMMIT get set by the call to add_version_info_from_vcs.
# DUMMY_VAR contains a version string which we don't care about.
add_version_info_from_vcs(DUMMY_VAR)

option(CLAD_BUILD_STATIC_ONLY "Does not build shared libraries. Useful when we have LLVM_ENABLE_PLUGINS=OFF (eg. Win or CYGWIN)" OFF)
if (NOT CLAD_BUILD_STATIC_ONLY AND NOT LLVM_ENABLE_PLUGINS)
  message(FATAL_ERROR "LLVM_ENABLE_PLUGINS is set to OFF. Please build clad with -DCLAD_BUILD_STATIC_ONLY=ON.")
endif()

add_subdirectory(lib)
add_subdirectory(tools)

# TODO: docs.
if (NOT CLAD_BUILD_STATIC_ONLY)
  add_subdirectory(test)
endif()

# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
if( CLAD_BUILT_STANDALONE AND MSVC_VERSION EQUAL 1600 )
  set(CLAD_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/clad.sln")
  if( EXISTS "${CLAD_SLN_FILENAME}" )
    file(APPEND "${CLAD_SLN_FILENAME}" "\n# This should be regenerated!\n")
  endif()
endif()
