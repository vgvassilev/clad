# Test runner infrastructure for Clad (copied and adapted from Clang). This
# configures the Clad test trees for use by Lit, and delegates to LLVM's lit
# test handlers.

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

# Needed for '%shlibext'
set(TARGET_SHLIBEXT "${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Define variables which cmake will export for lit. Typically in an installation
# they are not defined or defined in the wrong way.
if(CLAD_BUILT_STANDALONE)
  set(LLVM_SOURCE_DIR "")
  # cmake will fall back to LLVM-config (asking llvm-config --obj-root) and this
  # will return a build folder. This information will force lit to look for
  # utils such as FileCheck in /usr/local/... instead of the true location in
  # /usr/lib/llvm-*/
  set(LLVM_BINARY_DIR ${LLVM_INSTALL_PREFIX})
  set(LLVM_TOOLS_DIR ${LLVM_INSTALL_PREFIX})
  set(LLVM_LIBS_DIR ${LLVM_INSTALL_PREFIX}/lib/)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

option(CLANG_TEST_USE_VG "Run Clang tests under Valgrind" OFF)
if(CLANG_TEST_USE_VG)
  set(CLANG_TEST_EXTRA_ARGS ${CLANG_TEST_EXTRA_ARGS} "--vg")
endif ()

list(APPEND CLAD_TEST_DEPS clad cladCustomModelPlugin cladPrintModelPlugin)
# Try to append dependencies only if we are building in-tree.
if(NOT CLAD_BUILT_STANDALONE)
  list(APPEND CLAD_TEST_DEPS llvm-config FileCheck clang opt count not)
endif()

set(CLAD_TEST_PARAMS
  clad_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

# Unit tests
list(APPEND CLAD_TEST_DEPS CladUnitTests)

add_custom_target(clad-test-depends DEPENDS ${CLAD_TEST_DEPS})
set_target_properties(clad-test-depends PROPERTIES FOLDER "Clad tests")

# Set the main llvm source directory. Than can happen if
# CLAD_PATH_TO_LLVM_SOURCE is not set. This helps finding lit.py
if(NOT LLVM_MAIN_SRC_DIR)
  set(LLVM_MAIN_SRC_DIR ${LLVM_BUILD_MAIN_SRC_DIR})
endif()

add_lit_testsuite(check-clad "Running the Clad regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  LIT ${LIT_COMMAND}
  PARAMS ${CLAD_TEST_PARAMS}
  DEPENDS ${CLAD_TEST_DEPS}
  ARGS ${CLAD_TEST_EXTRA_ARGS} --verbose --succinct
  )
set_target_properties(check-clad PROPERTIES FOLDER "Clad tests")

add_lit_testsuites(CLAD ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${CLAD_TEST_PARAMS}
  DEPENDS ${CLAD_TEST_DEPS}
)
