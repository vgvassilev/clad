find_package(Torch REQUIRED)

CB_ADD_GBENCHMARK(LibTorchOpsBenchmark TorchOpsBenchmark.cpp)
target_link_libraries(LibTorchOpsBenchmark PRIVATE torch)

ADD_CLAD_EXECUTABLE(LibTorchOpsTest TorchOpsTest.cpp)
target_link_libraries(LibTorchOpsTest PRIVATE torch)
add_test(NAME clad-LibTorchOpsTest COMMAND LibTorchOpsTest)
set_tests_properties(clad-LibTorchOpsTest PROPERTIES
  LABELS "libtorch;benchmark"
  TIMEOUT 120)
