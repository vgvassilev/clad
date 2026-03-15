// RUN: %cladclang %s | FileCheck %s
#include<thread>

double f(double x){
      return x * x ; 
}

// CHECK: return 2 * x; 

