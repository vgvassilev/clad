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

list(APPEND CLAD_TEST_DEPS clad llvm-config FileCheck clang opt count not)

set(CLAD_TEST_PARAMS
  clad_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_custom_target(clad-test-depends DEPENDS ${CLAD_TEST_DEPS})
set_target_properties(clad-test-depends PROPERTIES FOLDER "Clad tests")

add_lit_testsuite(check-clad "Running the Clad regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  #LIT ${LLVM_LIT}
  PARAMS ${CLAD_TEST_PARAMS}
  DEPENDS ${CLAD_TEST_DEPS}
  ARGS ${CLAD_TEST_EXTRA_ARGS}
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
