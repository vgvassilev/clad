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

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

option(CLANG_TEST_USE_VG "Run Clang tests under Valgrind" OFF)
if(CLANG_TEST_USE_VG)
  set(CLANG_TEST_EXTRA_ARGS ${CLANG_TEST_EXTRA_ARGS} "--vg")
endif ()

list(APPEND CLAD_TEST_DEPS clad)
# Try to append dependencies only if we are building in-tree.
if(NOT CLAD_BUILT_STANDALONE)
  list(APPEND CLAD_TEST_DEPS llvm-config FileCheck clang opt count not)
endif()

set(CLAD_TEST_PARAMS
  clad_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_custom_target(clad-test-depends DEPENDS ${CLAD_TEST_DEPS})
set_target_properties(clad-test-depends PROPERTIES FOLDER "Clad tests")

# Set the main llvm source directory. Than can happen if
# CLAD_PATH_TO_LLVM_SOURCE is not set. This helps finding lit.py
if(NOT LLVM_MAIN_SRC_DIR)
  set(LLVM_MAIN_SRC_DIR ${LLVM_BUILD_MAIN_SRC_DIR})
endif()
add_lit_testsuite(check-clad "Running the Clad regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  #LIT ${LIT_COMMAND}
  PARAMS ${CLAD_TEST_PARAMS}
  DEPENDS ${CLAD_TEST_DEPS}
  ARGS ${CLAD_TEST_EXTRA_ARGS} --verbose --succinct
  )
set_target_properties(check-clad PROPERTIES FOLDER "Clad tests")

add_lit_testsuites(CLAD ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${CLAD_TEST_PARAMS}
  DEPENDS ${CLAD_TEST_DEPS}
)

# Add a legacy target spelling: clad-test
add_custom_target(clad-test)
add_dependencies(clad-test check-clad)
set_target_properties(clad-test PROPERTIES FOLDER "Clad tests")
