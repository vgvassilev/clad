add_clad_unittest(DifferentiatorHTests
  DifferentiatorH.cpp
  MultiIncludeDifferentiatorH.cpp
  )
add_test(NAME DifferentiatorHTests COMMAND ${CMAKE_BINARY_DIR}/unittests/Basic/DifferentiatorHTests)
