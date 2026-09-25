cmake_minimum_required(VERSION 3.7.0)

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(clad)
endif()

enable_language(CXX)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# clad.so links libcladDifferentiator.a; every target must be PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
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

# Define clad supported version of clang and llvm.
set(CLANG_MIN_SUPPORTED 14.0)
set(CLANG_MAX_SUPPORTED "23.2.x")
set(CLANG_VERSION_UPPER_BOUND 23.2.0)
set(LLVM_MIN_SUPPORTED 14.0)
set(LLVM_MAX_SUPPORTED "23.2.x")
set(LLVM_VERSION_UPPER_BOUND 23.2.0)

# If we are not building as a part of LLVM, build clad as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  set(CLAD_BUILT_STANDALONE 1)

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
    set(diag_kind FATAL_ERROR)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(diag_kind WARNING)
    endif()
    message(${diag_kind} "Found unsupported version: LLVM ${LLVM_PACKAGE_VERSION};\nPlease set LLVM_DIR pointing to the llvm version ${LLVM_MIN_SUPPORTED} to ${LLVM_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  ## Find supported Clang

  # Emscripten builds of clangInterpreter link lldWasm/lldCommon
  # (clang/lib/Interpreter/CMakeLists.txt), so ClangTargets.cmake references
  # LLD targets and find_package(Clang) fails unless LLD is found first.
  if (DEFINED LLD_DIR)
    find_package(LLD QUIET CONFIG PATHS ${LLD_DIR} "${LLD_DIR}/lib/cmake/lld" "${LLD_DIR}/cmake" NO_DEFAULT_PATH)
  else()
    find_package(LLD QUIET CONFIG HINTS "${LLVM_DIR}/../lld" "${LLVM_BINARY_DIR}/lib/cmake/lld")
  endif()

  if (DEFINED CLANG_VERSION)
    message(FATAL_ERROR "Clang's configuration is not versioned. Please set LLVM_VERSION instead.")
  endif(DEFINED CLANG_VERSION)

  # We have specified -DLLVM_VERSION and we must find matching clang. However,
  # ClangConfig.cmake is not versioned and we will need to work harder to find
  # the correct package.
  if (DEFINED LLVM_VERSION AND NOT DEFINED Clang_DIR)
    set(extra_hints HINTS ${Clang_DIR} "${LLVM_BINARY_DIR}/lib/cmake/clang" "${LLVM_BINARY_DIR}/cmake" "${LLVM_DIR}/../clang")
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

  if (LLVM_PACKAGE_VERSION VERSION_LESS CLANG_MIN_SUPPORTED OR LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL CLANG_VERSION_UPPER_BOUND)
    set(diag_kind FATAL_ERROR)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(diag_kind WARNING)
    endif()
    message(${diag_kind} "Found unsupported version: Clang ${LLVM_PACKAGE_VERSION};\nPlease set Clang_DIR pointing to the clang version ${CLANG_MIN_SUPPORTED} to ${CLANG_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: Clang ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

  if (CMAKE_CXX_STANDARD LESS 17)
    message(fatal "LLVM/Clad requires c++17 or later")
  endif()
  if (LLVM_VERSION_MAJOR GREATER_EQUAL 16)
    if (NOT CMAKE_CXX_STANDARD)
      set (CMAKE_CXX_STANDARD 17)
    endif()
    if (CMAKE_CXX_STANDARD LESS 17)
      message(fatal "LLVM/Clad requires C++17 or later")
    endif()
  else()
    if (NOT CMAKE_CXX_STANDARD)
      set (CMAKE_CXX_STANDARD 17)
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

  add_definitions(${LLVM_DEFINITIONS})

  # LLVM 22 sets _GLIBCXX_USE_CXX11_ABI twice (HandleLLVMOptions +
  # LLVMConfig), which trips -Werror=macro-redefined. Suppress that
  # specific warning -- it's an upstream duplication, not a code bug.
  if (LLVM_VERSION_MAJOR GREATER_EQUAL 22 AND NOT MSVC)
    add_compile_options(-Wno-macro-redefined)
  endif()

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

else()
  # Not standalone; we are building within llvm
  if (NOT "clang" IN_LIST LLVM_ENABLE_PROJECTS)
    message(FATAL_ERROR "Clang is not enabled; consider enabling it in LLVM_ENABLED_PROJECTS")
  endif()

  message(STATUS "Building Clad as part of LLVM")

  if (PACKAGE_VERSION VERSION_LESS CLANG_MIN_SUPPORTED OR PACKAGE_VERSION VERSION_GREATER_EQUAL CLANG_VERSION_UPPER_BOUND)
    message(WARNING "Building against unsupported version: Clang ${PACKAGE_VERSION};\n Supported versions are from ${CLANG_MIN_SUPPORTED} to ${CLANG_MAX_SUPPORTED}.")
  endif()

  set(CLANG_SOURCE_DIR ${LLVM_SOURCE_DIR}/../clang)
  set(CLANG_BINARY_DIR ${LLVM_TOOLS_BINARY_DIR})
  if (NOT CLANG_INCLUDE_DIRS)
    set(CLANG_INCLUDE_DIRS "${CLANG_SOURCE_DIR}/include")
  endif()
endif()

include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

## Code Coverage Configuration
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
  # The tables clad's own headers expand are rendered here, so an installed
  # clad needs them beside the headers that include them.
  PATTERN "*.def"
  )

install(FILES cmake/modules/AddClad.cmake
  DESTINATION lib/cmake/clad
  )

add_definitions( -D_GNU_SOURCE
  -DCLAD_SRCDIR_INCL="${CLAD_SOURCE_DIR}/include"
  -DCLAD_INSTDIR_INCL="${CLAD_BINARY_DIR}/include" )

option(CLAD_BUILD_STATIC_ONLY "Does not build shared libraries. Useful when we have LLVM_ENABLE_PLUGINS=OFF (eg. Win or CYGWIN)" OFF)

if (EMSCRIPTEN)
  # For the test executables and googletest only; the LLVM-style archive
  # targets append -fno-exceptions, which emcc >= 4 rejects alongside wasm-EH.
  set(CLAD_EXTRA_WASM_FLAGS "-fwasm-exceptions" CACHE STRING "Extra flags for wasm")
endif()
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

include(AddClad)

# Analyses.td and Options.td are where clad's analyses, the constructs they
# look for, the wording of every report and every option clad takes are
# written down. Everything rendered from them lands in the build directory and
# none of it is committed, so what clad compiles is what the .td files say and
# there is no second copy to go stale. The documentation pages are rendered the
# same way, by a target readthedocs runs before sphinx.
#
# A cross build cannot run a tool it built for the target, so it is given one
# built for the host: configure utils/clad-tblgen on its own against a native
# LLVM and pass the result as CLAD_TBLGEN. Without that there is nothing to
# render with, which is a configure error rather than a header that silently
# goes stale.
if (CLAD_TBLGEN)
  set(CLAD_TBLGEN_EXE ${CLAD_TBLGEN})
  set(CLAD_TBLGEN_EXE_FOR_LIT ${CLAD_TBLGEN})
  add_custom_target(clad-tblgen-ready)
elseif (CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR
    "Cross-compiling clad needs a clad-tblgen built for the host: configure "
    "utils/clad-tblgen against a native LLVM and pass -DCLAD_TBLGEN=<path>.")
else()
  # The TableGen library is looked up rather than taken from
  # LLVM_LINK_COMPONENTS, because with a libLLVM to link against llvm_config
  # folds every component into it and that library carries no TableGenMain.
  #
  # Only an in-tree build defines a real LLVMTableGen target. Standalone,
  # LLVMConfig exports an IMPORTED one of the same name, and taking that would
  # bring its include directories along with it.
  set(_clad_tablegen_imported TRUE)
  if (TARGET LLVMTableGen)
    get_target_property(_clad_tablegen_imported LLVMTableGen IMPORTED)
  endif()
  if (TARGET LLVMTableGen AND NOT _clad_tablegen_imported)
    set(CLAD_LLVM_TABLEGEN_LIB LLVMTableGen)
  else()
    find_library(CLAD_LLVM_TABLEGEN_LIB NAMES LLVMTableGen REQUIRED
                 HINTS ${LLVM_LIBRARY_DIR} ${LLVM_LIBRARY_DIRS})
  endif()
  add_subdirectory(utils/clad-tblgen)
  set(CLAD_TBLGEN_EXE $<TARGET_FILE:clad-tblgen>)
  # lit's site config is written at configure time, so it cannot carry a
  # generator expression; name where the tool lands. An in-tree build follows
  # LLVM's runtime directory, a standalone one the value set further up.
  if (LLVM_RUNTIME_OUTPUT_INTDIR)
    set(CLAD_TBLGEN_EXE_FOR_LIT ${LLVM_RUNTIME_OUTPUT_INTDIR}/clad-tblgen)
  else()
    set(CLAD_TBLGEN_EXE_FOR_LIT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/clad-tblgen)
  endif()
  add_custom_target(clad-tblgen-ready DEPENDS clad-tblgen)
endif()

# Public: clad's own headers expand them, so an installed clad needs them.
set(CLAD_GEN_INCL ${CMAKE_CURRENT_BINARY_DIR}/include/clad/Differentiator)
# Private: only the differentiator reads these, and nothing outside should.
set(CLAD_GEN_LIB ${CMAKE_CURRENT_BINARY_DIR}/lib/Differentiator)
file(MAKE_DIRECTORY ${CLAD_GEN_INCL} ${CLAD_GEN_LIB})

set(CLAD_OPTIONS_TD ${CMAKE_CURRENT_SOURCE_DIR}/lib/Differentiator/Options.td)
set(CLAD_ANALYSES_TD ${CMAKE_CURRENT_SOURCE_DIR}/lib/Differentiator/Analyses.td)
# Where an included .td is looked for, collected the way llvm's own
# TableGen.cmake collects it: from the include path, one -I each. Spelling it
# "-I ${LLVM_INCLUDE_DIRS}" is wrong in both directions. Against an LLVM build
# tree that variable names two directories, and the second is left to be read
# as a positional argument; in an in-tree build LLVMConfig is never read and it
# is empty, so the flag swallows the .td that follows it.
get_directory_property(CLAD_TBLGEN_INCLUDES INCLUDE_DIRECTORIES)
list(REMOVE_ITEM CLAD_TBLGEN_INCLUDES "")
list(REMOVE_DUPLICATES CLAD_TBLGEN_INCLUDES)
list(TRANSFORM CLAD_TBLGEN_INCLUDES PREPEND -I)

# add_custom_command rather than a target: these are inputs to compiling clad,
# so they have to be built before it and rebuilt when the table changes.
add_custom_command(
  OUTPUT ${CLAD_GEN_INCL}/Options.def
  COMMAND ${CLAD_TBLGEN_EXE} -gen-options ${CLAD_TBLGEN_INCLUDES}
          ${CLAD_OPTIONS_TD} -o ${CLAD_GEN_INCL}/Options.def
  DEPENDS clad-tblgen-ready ${CLAD_OPTIONS_TD}
  COMMENT "Rendering what Options.td describes"
  VERBATIM)

add_custom_command(
  OUTPUT ${CLAD_GEN_INCL}/Analyses.def ${CLAD_GEN_LIB}/AnalysisDescs.def
         ${CLAD_GEN_LIB}/Diagnostics.def
  COMMAND ${CLAD_TBLGEN_EXE} -gen-analyses ${CLAD_ANALYSES_TD}
          -o ${CLAD_GEN_INCL}/Analyses.def
  COMMAND ${CLAD_TBLGEN_EXE} -gen-analysis-descs ${CLAD_ANALYSES_TD}
          -o ${CLAD_GEN_LIB}/AnalysisDescs.def
  COMMAND ${CLAD_TBLGEN_EXE} -gen-diagnostics ${CLAD_ANALYSES_TD}
          -o ${CLAD_GEN_LIB}/Diagnostics.def
  DEPENDS clad-tblgen-ready ${CLAD_ANALYSES_TD}
  COMMENT "Rendering what Analyses.td describes"
  VERBATIM)

add_custom_target(clad-tables DEPENDS
  ${CLAD_GEN_INCL}/Options.def ${CLAD_GEN_INCL}/Analyses.def
  ${CLAD_GEN_LIB}/AnalysisDescs.def ${CLAD_GEN_LIB}/Diagnostics.def)

# The pages readthedocs builds; not committed either, and rendered only where
# the documentation is.
add_custom_target(clad-table-docs
  COMMAND ${CLAD_TBLGEN_EXE} -gen-options-docs ${CLAD_TBLGEN_INCLUDES}
          ${CLAD_OPTIONS_TD}
          -o ${CMAKE_CURRENT_SOURCE_DIR}/docs/userDocs/source/user/Options.rst
  COMMAND ${CLAD_TBLGEN_EXE} -gen-analyses-docs ${CLAD_ANALYSES_TD}
          -o ${CMAKE_CURRENT_SOURCE_DIR}/docs/userDocs/source/user/Analyses.rst
  COMMAND ${CLAD_TBLGEN_EXE} -gen-diagnostics-docs ${CLAD_ANALYSES_TD}
          -o ${CMAKE_CURRENT_SOURCE_DIR}/docs/userDocs/source/user/Diagnostics.rst
  DEPENDS clad-tblgen-ready ${CLAD_OPTIONS_TD} ${CLAD_ANALYSES_TD}
  COMMENT "Rendering the pages the tables describe"
  VERBATIM)

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
  if (CLAD_ENABLE_BENCHMARKS)
    include(GoogleBenchmark)
  endif(CLAD_ENABLE_BENCHMARKS)

  if (NOT CLAD_DISABLE_TESTS OR CLAD_ENABLE_BENCHMARKS)
    # Change the default compiler to the clang which we run clad upon. Our unittests
    # need to use a supported by clad compiler. Note that's a huge hack and it is
    # not guaranteed to work with cmake.
    set(stored_cxx_compiler ${CMAKE_CXX_COMPILER})
    set(stored_c_compiler ${CMAKE_C_COMPILER})
    set(stored_cxx_flags ${CMAKE_CXX_FLAGS})
    set(stored_c_flags ${CMAKE_C_FLAGS})
    # FIXME: Remove when vgvassilev/clad#1665 is resolved.
    disable_werror(CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
    # Swap both C and C++ -- clad_externalproject_add forwards both,
    # so a CXX-only swap leaves gtest with a mixed toolchain.
    set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
    set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
    # Drop GCC-only flags now that both C and CXX route through clang.
    foreach(_flags CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
      string(REPLACE "-fno-lifetime-dse" "" ${_flags} "${${_flags}}")
      string(REPLACE "-Wno-class-memaccess" "" ${_flags} "${${_flags}}")
    endforeach()
  endif()

  if (NOT CLAD_DISABLE_TESTS)
    add_subdirectory(unittests)
    add_subdirectory(test)
  endif()

  # Add benchmarking infrastructure.
  if (CLAD_ENABLE_BENCHMARKS)
    add_subdirectory(benchmark)
  endif(CLAD_ENABLE_BENCHMARKS)

  if (stored_cxx_compiler)
    # Restore the default compilers.
    set(CMAKE_CXX_COMPILER ${stored_cxx_compiler})
    set(CMAKE_C_COMPILER ${stored_c_compiler})
    set(CMAKE_CXX_FLAGS ${stored_cxx_flags})
    set(CMAKE_C_FLAGS ${stored_c_flags})
  endif()
elseif (EMSCRIPTEN AND NOT CLAD_DISABLE_TESTS)
  add_subdirectory(unittests)
endif()

# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
if( CLAD_BUILT_STANDALONE AND MSVC_VERSION EQUAL 1600 )
  set(CLAD_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/clad.sln")
  if( EXISTS "${CLAD_SLN_FILENAME}" )
    file(APPEND "${CLAD_SLN_FILENAME}" "\n# This should be regenerated!\n")
  endif()
endif()
