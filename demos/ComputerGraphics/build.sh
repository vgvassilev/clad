#!/bin/bash
clang -Ofast -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang /build/clad/build-clad-tracer/lib/clad.so -I../../include/ -I/usr/lib/gcc/x86_64-linux-gnu/10/include -x c++ -std=c++14 -lstdc++ -lm SmallPT-1.cpp -fopenmp=libiomp5 -o SmallPT-1 "$@"
