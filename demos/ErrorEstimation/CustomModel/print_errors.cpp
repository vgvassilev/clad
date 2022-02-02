//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo describing the usage of custom estimation models with the built -in
// error estimator of clad.
//
// author:  Garima Singh
//----------------------------------------------------------------------------//
// For information on how to run this demo, please take a look at the README

#include "clad/Differentiator/Differentiator.h"

#include <iostream> // For std::cout

// Use a trivial dummy example to check if error printing takes place correctly.
// Essentially sums two values together and then returns a square of the sum.
// (i.e (x + y) * (x + y) )
float func(float x, float y) {
  float mul, sum;
  sum = x + y;
  mul = sum * sum;
  return mul;
}

int main() {
  // Call error-estimate on func, this time with a template parameter to signal
  // that error printing is enabled. As demonstared in ./test.cpp, you can omit
  // this parameter to disable error printing.
  auto df = clad::estimate_error</*PrintErrors=*/true>(func);

  // The next step is to actually define some variables we will use to execute
  // 'df' as created above.
  float dx = 0, dy = 0;
  // Declare some double values eventhough our function 'func' takes in float
  // values. This is so that we can compare errors later.
  double x = 9.999E-4, y = 0.001E-4, final_error = 0;

  // Now, we can actually first dump the code to analyse if the print code
  // generation is correct.
  df.dump();

  // Now, we can finally run the 'df' function via 'execute'. Notice, we pass an
  // addition argument std::cout. Clad actually prints to whatever stream you
  // pass it.
  df.execute(x, y, &dx, &dy, final_error, std::cout);
  // Note: you can also pass the error printing to a file by either doing so in
  // code:
  //   std::ofstream myfile("my_fp_error_file");
  //   df.execute(x, y, &dx, &dy, final_error, myfile);
  // Or by piping output from the compilation command of this demo to a file:
  // $ --my compile command-- > my_fp_error_file.txt

  // Now let us compare the actual errors and the ones printed on the console.
  // This is just to showcase the error values calculated by clad and the actual
  // error values.
  double dbl_sum, dbl_mul;
  float sum, mul, flt_x = x, flt_y = y;
  // Now we emulate the function.
  dbl_sum = x + y;
  sum = flt_x + flt_y;

  dbl_mul = dbl_sum * dbl_sum;
  mul = sum * sum;

  // Calculate the errors.
  double sum_err = std::abs(dbl_sum - sum);
  double mul_err = std::abs(dbl_mul - mul);
  double var_err_x = std::abs(x - flt_x);
  double var_err_y = std::abs(y - flt_y);

  // Print them for comparision.
  std::cout << "Sum Error: " << sum_err << " Mul Error: " << mul_err
            << " var error x: " << var_err_x << "var error y: " << var_err_y;
}
