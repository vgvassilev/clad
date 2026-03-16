// RUN: %cladclang %s -I%S/..//../include | FileCheck %s

#include<thread>
#include "clad/Differentiator/Differentiator.h"

double f(double x){
      return x * x; 
}

int main() {
   auto df = clad::differentiate(f, "x");
   df.execute(3);
}

// CHECK: return _d_x * x + x * _d_x; 

