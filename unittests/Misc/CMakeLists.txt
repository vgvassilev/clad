add_clad_unittest(MiscTests
  main.cpp
  CallDeclOnly.cpp
  Defs.cpp
  DynamicGraph.cpp
)

# Create a library from Defs.cpp
add_library(Defs SHARED Defs.cpp)
target_include_directories(Defs PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(Defs PUBLIC stdc++ pthread m)

# Link the library to the test
target_link_libraries(MiscTests PRIVATE Defs)
