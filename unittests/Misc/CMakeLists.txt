add_clad_unittest(MiscTests
  main.cpp
  CallDeclOnly.cpp
  Defs.cpp
  DynamicGraph.cpp
)

# Create a library from the Defs.cpp file
ADD_CLAD_LIBRARY(Defs Defs.cpp)

# Link the library to the test
target_link_libraries(MiscTests PRIVATE Defs)
