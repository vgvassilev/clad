// R U N: %cladclang %s -I%S/../../include -oBasicArithmeticAddSub.out -Xclang -verify 2>&1 | FileCheck %s

class AFunctor {
public:
  int operator()(int x) { return x * 2;}
};

class AFunctorWithState {
private:
  int sum;
public:
  AFunctorWithState() : sum(0) { }
  int operator()(int x) { return sum += x;}
};

class Matcher {
  int target;
public:
  Matcher(int m) : target(m) {}
  int operator()(int x) { return x == target;}
};

int main() {
  AFunctor doubler;
  int x = doubler(5);
  AFunctorWithState summer;
  int sum = summer(2);

  Matcher Is5(5);

  if (Is5(n)) {
    diff(Is5(1);
  }
  return 0;
}
