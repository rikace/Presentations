using System;

namespace CSMisc
{
	public static class RetryExtensions
	{
		public static Func<TResult> Partial<TParam1, TResult>(
			this Func<TParam1, TResult> func, TParam1 parameter)
		{
			// return a function that wraps the execution
			// of the func passed as parameter and passing the parameter expected 
			// as result with return Func of TResult
			return () => func(parameter); 
		}

		public static Func<TParam1, Func<TResult>> Curry<TParam1, TResult>
		(this Func<TParam1, TResult> func)
		{
			return parameter => () => func(parameter);
		}
	}
}