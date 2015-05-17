using System;
using System.Linq;
using System.Collections.Generic;

namespace CSMisc
{
	public class NumberClassifierFunctional
	{
		public static IEnumerable<int> FactorsOf(int number) {

			return Enumerable.Range(1, number + 1)
				.Where(potential => number % potential == 0);
	}

		public static int Sum(int number) {
			return FactorsOf(number).Sum() - number;
	}

		public static bool IsPerfect(int number) {
			return Sum(number) == number;
	}

		public static bool IsAbundant(int number) {
			return Sum(number)> number;
	}

		public static bool IsDeficient(int number) {
			return Sum(number) < number;
		}
	}
}