// RUN: %cladclang %s -I%S/../../include -oCladMatrix.out 2>&1
// RUN: ./CladMatrix.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

int main() {
  clad::matrix<int> test_mat(2, 2);
  for (int i = 0; i < test_mat.rows(); i++) {
    for (int j = 0; j < test_mat.cols(); ++j) {
      printf("%d, %d : %d\n", i, j, test_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 0
  // CHECK-EXEC: 0, 1 : 0
  // CHECK-EXEC: 1, 0 : 0
  // CHECK-EXEC: 1, 1 : 0

  clad::matrix<int> test_res = test_mat + 2*clad::identity_matrix<int>(2, 2);
  for (int i = 0; i < test_res.rows(); i++) {
    for (int j = 0; j < test_res.cols(); ++j) {
      printf("%d, %d : %d\n", i, j, test_res(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 2
  // CHECK-EXEC: 0, 1 : 0
  // CHECK-EXEC: 1, 0 : 0
  // CHECK-EXEC: 1, 1 : 2

  for (int i = 0; i < test_mat.rows(); i++) {
    test_mat(i, 1) = 2;
  }
  for (int i = 0; i < test_mat.rows(); i++) {
    for (int j = 0; j < test_mat.cols(); ++j) {
      printf("%d, %d : %d\n", i, j, test_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 0
  // CHECK-EXEC: 0, 1 : 2
  // CHECK-EXEC: 1, 0 : 0
  // CHECK-EXEC: 1, 1 : 2

  clad::matrix<double> test2_mat = clad::identity_matrix<double>(3, 2);
  for (int i = 0; i < test2_mat.rows(); i++) {
    for (int j = 0; j < test2_mat.cols(); ++j) {
      printf("%d, %d : %.2f\n", i, j, test2_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 1.00
  // CHECK-EXEC: 0, 1 : 0.00
  // CHECK-EXEC: 1, 0 : 0.00
  // CHECK-EXEC: 1, 1 : 1.00
  // CHECK-EXEC: 2, 0 : 0.00
  // CHECK-EXEC: 2, 1 : 0.00

  // To each row of test2_mat, add a newly created clad array.
  clad::array<int> arr = {1, 2};
  for (int i = 0; i < test2_mat.rows(); i++) {
    test2_mat[i] += arr;
  }
  for (int i = 0; i < test2_mat.rows(); i++) {
    for (int j = 0; j < test2_mat.cols(); ++j) {
      printf("%d, %d : %.2f\n", i, j, test2_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 2.00
  // CHECK-EXEC: 0, 1 : 2.00
  // CHECK-EXEC: 1, 0 : 1.00
  // CHECK-EXEC: 1, 1 : 3.00
  // CHECK-EXEC: 2, 0 : 1.00
  // CHECK-EXEC: 2, 1 : 2.00

  // From each row of test2_mat, subtract back the same array.
  for (int i = 0; i < test2_mat.rows(); i++) {
    test2_mat[i] -= arr;
  }
  for (int i = 0; i < test2_mat.rows(); i++) {
    for (int j = 0; j < test2_mat.cols(); ++j) {
      printf("%d, %d : %.2f\n", i, j, test2_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 1.00
  // CHECK-EXEC: 0, 1 : 0.00
  // CHECK-EXEC: 1, 0 : 0.00
  // CHECK-EXEC: 1, 1 : 1.00
  // CHECK-EXEC: 2, 0 : 0.00
  // CHECK-EXEC: 2, 1 : 0.00

  // Multiply each row of test2_mat by 2.
  for (int i = 0; i < test2_mat.rows(); i++) {
    test2_mat[i] *= 2;
  }
  for (int i = 0; i < test2_mat.rows(); i++) {
    for (int j = 0; j < test2_mat.cols(); ++j) {
      printf("%d, %d : %.2f\n", i, j, test2_mat(i, j));
    }
  }

  // CHECK-EXEC: 0, 0 : 2.00
  // CHECK-EXEC: 0, 1 : 0.00
  // CHECK-EXEC: 1, 0 : 0.00
  // CHECK-EXEC: 1, 1 : 2.00
  // CHECK-EXEC: 2, 0 : 0.00
  // CHECK-EXEC: 2, 1 : 0.00

  // Divide each row of test2_mat by 4.
  for (int i = 0; i < test2_mat.rows(); i++) {
    test2_mat[i] /= 4;
  }
  for (int i = 0; i < test2_mat.rows(); i++) {
    for (int j = 0; j < test2_mat.cols(); ++j) {
      printf("%d, %d : %.2f\n", i, j, test2_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 0.50
  // CHECK-EXEC: 0, 1 : 0.00
  // CHECK-EXEC: 1, 0 : 0.00
  // CHECK-EXEC: 1, 1 : 0.50
  // CHECK-EXEC: 2, 0 : 0.00
  // CHECK-EXEC: 2, 1 : 0.00

  // Generate an identity matrix of size 3x3, with diagonal offset 1.
  clad::matrix<int> test3_mat = clad::identity_matrix<int>(3, 3, 1);
  for (int i = 0; i < test3_mat.rows(); i++) {
    for (int j = 0; j < test3_mat.cols(); ++j) {
      printf("%d, %d : %d\n", i, j, test3_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 0
  // CHECK-EXEC: 0, 1 : 1
  // CHECK-EXEC: 0, 2 : 0
  // CHECK-EXEC: 1, 0 : 0
  // CHECK-EXEC: 1, 1 : 0
  // CHECK-EXEC: 1, 2 : 1
  // CHECK-EXEC: 2, 0 : 0
  // CHECK-EXEC: 2, 1 : 0
  // CHECK-EXEC: 2, 2 : 0

  // Change row 0 of test3_mat to {1, 2, 3}.
  clad::array<int> arr2 = {1, 2, 3};
  test3_mat[0] = arr2;
  for (int i = 0; i < test3_mat.rows(); i++) {
    for (int j = 0; j < test3_mat.cols(); ++j) {
      printf("%d, %d : %d\n", i, j, test3_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 1
  // CHECK-EXEC: 0, 1 : 2
  // CHECK-EXEC: 0, 2 : 3
  // CHECK-EXEC: 1, 0 : 0
  // CHECK-EXEC: 1, 1 : 0
  // CHECK-EXEC: 1, 2 : 1
  // CHECK-EXEC: 2, 0 : 0
  // CHECK-EXEC: 2, 1 : 0
  // CHECK-EXEC: 2, 2 : 0

  test3_mat[0][0] = 4;
  for (int i = 0; i < test3_mat.rows(); i++) {
    for (int j = 0; j < test3_mat.cols(); ++j) {
      printf("%d, %d : %d\n", i, j, test3_mat(i, j));
    }
  }
  // CHECK-EXEC: 0, 0 : 4
  // CHECK-EXEC: 0, 1 : 2
  // CHECK-EXEC: 0, 2 : 3
  // CHECK-EXEC: 1, 0 : 0
  // CHECK-EXEC: 1, 1 : 0
  // CHECK-EXEC: 1, 2 : 1
  // CHECK-EXEC: 2, 0 : 0
  // CHECK-EXEC: 2, 1 : 0
  // CHECK-EXEC: 2, 2 : 0
}
