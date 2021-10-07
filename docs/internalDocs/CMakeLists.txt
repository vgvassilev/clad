
find_package(Doxygen REQUIRED)

set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in)
set(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg)

set(docs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
set(docs_builddir ${CMAKE_CURRENT_BINARY_DIR})
set(clad_srcdir ${CMAKE_SOURCE_DIR})
file(READ ${CMAKE_SOURCE_DIR}/VERSION PACKAGE_VERSION)

configure_file(${DOXYFILE_IN} ${DOXYFILE} @ONLY)

add_custom_target(doxygen-clad
                  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                  COMMENT "Generate clad documentation with Doxygen"
                  VERBATIM)


