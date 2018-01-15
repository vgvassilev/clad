# If we are not building as a part of LLVM, build clad as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(clad)
  cmake_minimum_required(VERSION 3.4.3)
  find_package(Subversion)

  set(CLAD_PATH_TO_LLVM_SOURCE "" CACHE PATH
    "Path to LLVM source code. Not necessary if using an installed LLVM.")
  set(CLAD_PATH_TO_LLVM_BUILD "" CACHE PATH
    "Path to the directory where LLVM was built or installed.")

  if( CLAD_PATH_TO_LLVM_SOURCE )
    if( NOT EXISTS "${CLAD_PATH_TO_LLVM_SOURCE}/cmake/config-ix.cmake" )
      message(FATAL_ERROR "Please set CLAD_PATH_TO_LLVM_SOURCE to the root directory of LLVM source code.")
    else()
      get_filename_component(LLVM_MAIN_SRC_DIR ${CLAD_PATH_TO_LLVM_SOURCE}
        ABSOLUTE)
      list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
    endif()
  endif()

  if( NOT EXISTS "${CLAD_PATH_TO_LLVM_BUILD}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}" )
    message(FATAL_ERROR "Please set CLAD_PATH_TO_LLVM_BUILD to a directory containing a LLVM build.")
  endif()

  # In case this was a path to a build folder of llvm still try to find AddLLVM.cmake
  list(APPEND CMAKE_MODULE_PATH "${CLAD_PATH_TO_LLVM_BUILD}/lib/cmake/llvm/")
  list(APPEND CMAKE_MODULE_PATH "${CLAD_PATH_TO_LLVM_BUILD}/share/llvm/cmake")

  get_filename_component(PATH_TO_LLVM_BUILD ${CLAD_PATH_TO_LLVM_BUILD} ABSOLUTE)

  include("${PATH_TO_LLVM_BUILD}/lib/cmake/llvm/LLVMConfig.cmake")
  include(AddLLVM)
  include(HandleLLVMOptions)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  if(NOT LLVM_BINARY_DIR)
    if(LLVM_BUILD_BINARY_DIR)
      set(LLVM_BINARY_DIR ${LLVM_BUILD_BINARY_DIR})
    else()
      set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})
    endif()
  endif()

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${LLVM_BINARY_DIR}/include/")
  # If we found the corresponding sources, add them as an include path.
  if (LLVM_BUILD_MAIN_INCLUDE_DIR)
    include_directories("${LLVM_BUILD_MAIN_INCLUDE_DIR}")
    # FIXME: This expects clang to be hardcoded in tools/clang. LLVM supports
    # building clang out of tree.
    include_directories("${LLVM_BINARY_DIR}/tools/clang/include" "${LLVM_BUILD_MAIN_SRC_DIR}/tools/clang/include/")
  endif()
  link_directories("${LLVM_LIBRARY_DIRS}")

  set(LLVM_TABLEGEN_EXE "${PATH_TO_LLVM_BUILD}/bin/llvm-tblgen")

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

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

add_subdirectory(lib)
add_subdirectory(tools)

# TODO: docs.
add_subdirectory(test)

# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
if( CLAD_BUILT_STANDALONE AND MSVC_VERSION EQUAL 1600 )
  set(CLAD_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/clad.sln")
  if( EXISTS "${CLAD_SLN_FILENAME}" )
    file(APPEND "${CLAD_SLN_FILENAME}" "\n# This should be regenerated!\n")
  endif()
endif()
