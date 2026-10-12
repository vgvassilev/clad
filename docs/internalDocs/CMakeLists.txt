
find_package(Doxygen REQUIRED)

set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in)
set(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg)

set(docs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
set(docs_builddir ${CMAKE_CURRENT_BINARY_DIR})
# Ours even when clad is part of a bigger build, where CMAKE_SOURCE_DIR
# and CMAKE_BINARY_DIR belong to whoever is on top.
set(clad_srcdir ${CLAD_SOURCE_DIR})
set(clad_bindir ${CLAD_BINARY_DIR})
file(READ ${clad_srcdir}/VERSION PACKAGE_VERSION)

configure_file(${DOXYFILE_IN} ${DOXYFILE} @ONLY)

add_custom_target(doxygen-clad
                  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
                  WORKING_DIRECTORY ${clad_bindir}
                  COMMENT "Generate clad documentation with Doxygen"
                  VERBATIM)


