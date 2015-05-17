using System;
using System.Collections.Generic;

namespace CSMisc
{
	public static class MemoizeTest
	{
		public static Func<A, R> Memoize<A, R>(this Func<A, R> f)
		{
			var map = new Dictionary<A, R>();
			return a =>
			{
				R value;
				if (map.TryGetValue(a, out value))
					return value;
				value = f(a);
				map.Add(a, value);
				return value;
			};
		}

		static int Fibonacci(int value)
		{
			Func<int, int> fib = null;
			fib = n => n > 1 ? fib(n - 1) + fib(n - 2) : n;
			int result = fib (value);
            return result;
		}

		public static int TestFibMem()
		{
            //Func<int, int> fib = null;
            //fib = n => n > 1 ? fib(n - 1) + fib(n - 2) : n;
            //var fibonacciMemoized = fib.Memoize();
            //var result = fibonacciMemoized(35);
            //return result;

            Func<int, int> fibonacci = Fibonacci;
            var fibonacciMemoized = fibonacci.Memoize();
           
            var result = fibonacciMemoized(35);
            result = fibonacciMemoized(35);
            result = fibonacciMemoized(35);
            result = fibonacciMemoized(35);
            result = fibonacciMemoized(35);
            result = fibonacciMemoized(35);
            result = fibonacciMemoized(35);
            result = fibonacciMemoized(35);
            return result;
		}

		public static int TestFib()
		{
			Func<int, int> fibonacci = Fibonacci;  
			var result = fibonacci(35);
			result = fibonacci(35);
			result = fibonacci(35);
            //result = fibonacci(35);
            //result = fibonacci(35);
			return result;
		}
	}
}

