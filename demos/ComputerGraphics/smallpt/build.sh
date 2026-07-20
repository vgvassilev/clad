#!/bin/bash
clang-18 -Ofast -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang ../../../inst/lib/clad.so -I../../../include/ -I/usr/lib/gcc/x86_64-linux-gnu/10/include -x c++ -std=c++14 -lstdc++ -lm SmallPT.cpp -fopenmp=libiomp5 -o SmallPT "$@"
