#!/bin/bash
clang-18 -Ofast -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang ../../../inst/lib/clad.so -I../../../include/ -x c++ -std=c++14 -lstdc++ -lm cpp-smallpt.cpp -fopenmp=libiomp5 -o cpp-smallpt "$@"
clang-18 -Ofast -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang ../../../inst/lib/clad.so -I../../../include/ -x c++ -std=c++14 -lstdc++ -lm diff-tracer-1.cpp -fopenmp=libiomp5 -o diff-tracer-1 "$@"
