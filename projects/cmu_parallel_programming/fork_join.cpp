#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;


long fib_seq (long n) {
  long result;
  if (n < 2) {
    result = n;
  } else {
    long a, b;
    a = fib_seq(n-1);
    b = fib_seq(n-2);
    result = a + b;
  }
  return result;
}


int main()
{
    long n = 10;

    cout << "The Fib num for n = " << n << " would be: " << fib_seq(n);
}