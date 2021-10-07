list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CreateSphinxTarget)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/source/conf.py 
               ${CMAKE_CURRENT_BINARY_DIR}/source/conf.py
               @ONLY
              )

create_sphinx_target(
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source
  TARGET_NAME sphinx-clad
)