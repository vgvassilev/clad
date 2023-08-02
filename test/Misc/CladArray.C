// RUN: %cladclang %s -I%S/../../include -oCladArray.out 2>&1
// RUN: ./CladArray.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

int main() {
  clad::array<int> test_arr(3);
  clad::array<int> clad_arr(3);
  clad::array<double> double_test_arr(3);
  int arr[] = {2, 2, 2};
  clad::array_ref<int> arr_ref(arr, 3);

  for (int i = 0; i < 3; i++) {
    test_arr[i] = 0;
    clad_arr[i] = i + 1;
    double_test_arr[i] = 0;
  }

  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 0
  //CHECK-EXEC: 1 : 0
  //CHECK-EXEC: 2 : 0

  test_arr = clad_arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 1
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 3

  test_arr += clad_arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 2
  //CHECK-EXEC: 1 : 4
  //CHECK-EXEC: 2 : 6

  test_arr -= clad_arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 1
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 3

  test_arr *= clad_arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 1
  //CHECK-EXEC: 1 : 4
  //CHECK-EXEC: 2 : 9

  test_arr /= clad_arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 1
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 3

  test_arr += arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 3
  //CHECK-EXEC: 1 : 4
  //CHECK-EXEC: 2 : 5

  test_arr -= arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 1
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 3

  test_arr *= arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 2
  //CHECK-EXEC: 1 : 4
  //CHECK-EXEC: 2 : 6

  test_arr /= arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 1
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 3

  test_arr = arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 2
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 2

  test_arr = clad_arr;

  test_arr += arr_ref;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 3
  //CHECK-EXEC: 1 : 4
  //CHECK-EXEC: 2 : 5

  test_arr -= arr_ref;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 1
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 3

  test_arr *= arr_ref;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 2
  //CHECK-EXEC: 1 : 4
  //CHECK-EXEC: 2 : 6

  test_arr /= arr_ref;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 1
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 3

  test_arr = arr_ref;
  for (int i = 0; i < 3; i++) {
    printf("%d : %d\n", i, test_arr[i]);
  }
  //CHECK-EXEC: 0 : 2
  //CHECK-EXEC: 1 : 2
  //CHECK-EXEC: 2 : 2

  double_test_arr += 0;
  for (int i = 0; i < 3; i++) {
    printf("%d : %.2f\n", i, double_test_arr[i]);
  }
  //CHECK-EXEC: 0 : 0.00
  //CHECK-EXEC: 1 : 0.00
  //CHECK-EXEC: 2 : 0.00

  double_test_arr += test_arr;
  for (int i = 0; i < 3; i++) {
    printf("%d : %.2f\n", i, double_test_arr[i]);
  }
  // CHECK-EXEC: 0 : 2.00
  // CHECK-EXEC: 1 : 2.00
  // CHECK-EXEC: 2 : 2.00

  clad::array<double> double_test_arr2 = clad::one_hot_vector<double> (3, 1);
  for (int i = 0; i < 3; i++) {
    printf("%d : %.2f\n", i, double_test_arr2[i]);
  }
  // CHECK-EXEC: 0 : 0.00
  // CHECK-EXEC: 1 : 1.00
  // CHECK-EXEC: 2 : 0.00

  // Create a slice of double_test_arr2 and modify one of its elements.
  // This should not modify the original array.
  clad::array<double> double_test_arr2_slice = double_test_arr2.slice(1, 2);
  double_test_arr2_slice[0] = 2;
  for (int i = 0; i < 3; i++) {
    printf("%d : %.2f\n", i, double_test_arr2[i]);
  }
  // CHECK-EXEC: 0 : 0.00
  // CHECK-EXEC: 1 : 1.00
  // CHECK-EXEC: 2 : 0.00

  // Create a slice by creating array_ref and modify one of its elements.
  // This should modify the original array.
  clad::array_ref<double> ref_slice = clad::array_ref<double>(double_test_arr2).slice(1, 2);
  ref_slice[0] = 2;
  for (int i = 0; i < 3; i++) {
    printf("%d : %.2f\n", i, double_test_arr2[i]);
  }
  // CHECK-EXEC: 0 : 0.00
  // CHECK-EXEC: 1 : 2.00
  // CHECK-EXEC: 2 : 0.00

  // Create a clad array from pointer and modify one of its elements.
  // This should not modify the original array.
  double double_test_arr3[] = {1, 2, 3};
  clad::array<double> double_test_arr4(double_test_arr3, 3);
  double_test_arr4[0] = 2;
  for (int i = 0; i < 3; i++) {
    printf("%d : %.2f\n", i, double_test_arr3[i]);
  }
  // CHECK-EXEC: 0 : 1.00
  // CHECK-EXEC: 1 : 2.00
  // CHECK-EXEC: 2 : 3.00
}
