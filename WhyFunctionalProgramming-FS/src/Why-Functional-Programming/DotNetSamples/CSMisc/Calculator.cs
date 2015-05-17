using System;

namespace CSMisc
{
	interface ICalculator 
	{
		int Calculate(int input);
	}


	class Adding3Calculator: ICalculator
	{
		public int Calculate(int input) { return input + 1; }
	}


	class LoggingCalculator: ICalculator
	{
		ICalculator _innerCalculator;

		LoggingCalculator(ICalculator innerCalculator)
		{
			_innerCalculator = innerCalculator;
		}

		public int Calculate(int input) 
		{ 
			Console.WriteLine("input is {0}", input);
			var result  = _innerCalculator.Calculate(input);
			Console.WriteLine("result is {0}", result);
			return result; 
		}
	}
}
