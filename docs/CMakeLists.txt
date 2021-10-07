if (CLAD_ENABLE_DOXYGEN)
  add_subdirectory(internalDocs)
endif()

if (CLAD_ENABLE_SPHINX)
  add_subdirectory(userDocs)
endif()

if(NOT CLAD_ENABLE_DOXYGEN AND NOT CLAD_ENABLE_SPHINX)
  message(WARNING "No documentation tool enabled. Enable any documentation tool "
                  "to generate documentation build targets")
endif()