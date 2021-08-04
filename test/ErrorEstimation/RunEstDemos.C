// RUN: %cladclang -x c++ -lm -lstdc++ %S/../../demos/ErrorEstimation/FloatSum.cpp -I%S/../../include 2>&1
//CHECK_FLOAT_SUM-NOT: {{.*error|warning|note:.*}}

//-----------------------------------------------------------------------------/
//  Demo: CustomModel
//-----------------------------------------------------------------------------/

// TODO
