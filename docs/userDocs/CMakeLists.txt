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

# The pages listing the analyses, what clad says and the options are rendered
# from the tables, so they have to exist before sphinx reads the directory.
add_dependencies(sphinx-clad clad-options clad-analyses)
