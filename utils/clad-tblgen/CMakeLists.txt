# Configurable on its own, not only as part of clad. A cross build cannot run
# a tool it compiled for the target, so it needs one built for the host, and
# building the host one must not drag in everything clad needs -- this tool
# reads a table and writes text, and wants nothing from clang.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(clad-tblgen C CXX)
  # The same standard clad's own build asks for; LLVM's headers need it.
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  find_package(LLVM REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
  include(HandleLLVMOptions)
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  find_library(CLAD_LLVM_TABLEGEN_LIB NAMES LLVMTableGen REQUIRED
               HINTS ${LLVM_LIBRARY_DIR} ${LLVM_LIBRARY_DIRS})
endif()

# Let LLVM say how it wants to be linked; the answer decides which libraries
# appear and in what order.
#
# With a libLLVM to link against, llvm_config folds every component into it.
# That library carries no TableGenMain, so the one that was looked up has to be
# named as well -- and a shared library ahead of it keeps its symbols
# available whatever the order.
#
# Without one the components are archives, and a linker walking archives only
# looks forward: naming TableGen is enough, because LLVMExports gives it an
# INTERFACE_LINK_LIBRARIES of LLVMSupport and cmake puts Support after it.
if (LLVM_LINK_LLVM_DYLIB)
  set(LLVM_LINK_COMPONENTS Support)
  add_llvm_executable(clad-tblgen TableGen.cpp)
  target_link_libraries(clad-tblgen PRIVATE ${CLAD_LLVM_TABLEGEN_LIB})
else()
  set(LLVM_LINK_COMPONENTS TableGen)
  add_llvm_executable(clad-tblgen TableGen.cpp)
endif()
set_target_properties(clad-tblgen PROPERTIES FOLDER "Clad tools")
