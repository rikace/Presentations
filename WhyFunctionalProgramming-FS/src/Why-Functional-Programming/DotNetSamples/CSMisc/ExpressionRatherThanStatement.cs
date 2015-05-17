using System;

namespace CSMisc
{
	public class ExpressionRatherThanStatement
	{
		public void Test ()
		{
			// Statements don't return values, 
			// so you have to use temporary variables that 
			// are assigned to from within statement bodies

			bool executeStatement = false;

			int result = 0;     
			if (executeStatement)
			{
				result = 42; 
			}
			Console.WriteLine("result={0}", result);

			/*
			Because the if-then block is a statement, 
			the result variable must be defined outside the statement 
			but assigned to from inside the statement, which leads to issues such as:

			-	What initial value should result be set to?
			-	What if I forget to assign to the result variable?
			-	What is the value of the result variable in the "else" case?
			*/

			// same code, rewritten in an expression-oriented style:

			result = (false) ? 42 : 0;
			Console.WriteLine("result={0}", result);


			//	The result variable is declared at the same time that it is assigned. 

			//	No variables have to be set up "outside" the expression and there 
			//	is no worry about what initial value they should be set to.

			//	The "else" is explicitly handled. 
			//	There is no chance of forgetting to do an assignment in one of the branches.

			//	It is not possible to forget to assign result, 
			//	because then the variable would not even exist!	
		}
	}
}

