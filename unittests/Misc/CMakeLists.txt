add_clad_unittest(MiscTests
  CallDeclOnly.cpp
  Defs.cpp
  DynamicGraph.cpp
)
add_test(NAME MiscTests COMMAND ${CMAKE_BINARY_DIR}/unittests/Misc/MiscTests)

# Create a library from the Defs.cpp file
ADD_CLAD_LIBRARY(Defs Defs.cpp)
# Link the library to the test
target_link_libraries(MiscTests PRIVATE Defs)
