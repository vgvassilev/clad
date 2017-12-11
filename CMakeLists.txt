# If we are not building as a part of LLVM, build clad as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(clad)
  cmake_minimum_required(VERSION 2.8)
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

  list(APPEND CMAKE_MODULE_PATH "${CLAD_PATH_TO_LLVM_BUILD}/share/llvm/cmake")

  get_filename_component(PATH_TO_LLVM_BUILD ${CLAD_PATH_TO_LLVM_BUILD}
    ABSOLUTE)

  include(AddLLVM)
  include("${CLAD_PATH_TO_LLVM_BUILD}/share/llvm/cmake/LLVMConfig.cmake")
  include(HandleLLVMOptions)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${PATH_TO_LLVM_BUILD}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  link_directories("${PATH_TO_LLVM_BUILD}/lib")

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

macro(add_clad_library name)
  llvm_process_sources(srcs ${ARGN})
  if(MSVC_IDE OR XCODE)
    string( REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list( GET split_path -1 dir)
    file( GLOB_RECURSE headers
      ../../include/clad${dir}/*.h
      ../../include/clad${dir}/*.td
      ../../include/clad${dir}/*.def)
    set(srcs ${srcs} ${headers})
  endif(MSVC_IDE OR XCODE)
  if (MODULE)
    set(libkind MODULE)
  elseif (SHARED_LIBRARY)
    set(libkind SHARED)
  else()
    set(libkind)
  endif()
  add_library( ${name} ${libkind} ${srcs} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )

  target_link_libraries( ${name} ${LLVM_USED_LIBS} )
  llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
  target_link_libraries( ${name} ${LLVM_COMMON_LIBS} )

  if(MSVC)
    get_target_property(cflag ${name} COMPILE_FLAGS)
    if(NOT cflag)
      set(cflag "")
    endif(NOT cflag)
    set(cflag "${cflag} /Za")
    set_target_properties(${name} PROPERTIES COMPILE_FLAGS ${cflag})
  endif(MSVC)
  install(TARGETS ${name}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
  set_target_properties(${name} PROPERTIES FOLDER "clad libraries")
endmacro(add_clad_library)

macro(add_clad_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "clad executables")
endmacro(add_clad_executable)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/../clang/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../clang/include
  )

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
